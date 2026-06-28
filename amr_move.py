import rclpy
from rclpy.node import Node
import json
import math 
import time

from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import TaskResult
from std_msgs.msg import String
from nav_msgs.msg import Path
from transforms3d.euler import euler2quat


from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from geometry_msgs.msg import PoseWithCovarianceStamped

# INITIAL_X = 0.0
# INITIAL_Y = 0.11
# INITIAL_YAW = 0.0

# PATROL TARGET
PATROL_TARGETS = [
            (-0.805, 0.392, 0.0),          
            (-3.6, 0.265, 0.0),
            (-5.0, 0.0391, 0.0),
            (-4.72, 3.74, 0.0),
            (-0.103, 3.58, 0.0)

        ]

# amr_move.py (집 테스트용 임시 좌표 - 로봇 기준 앞뒤양옆 1~2m)
# PATROL_TARGETS = [
#     (1.0, 0.0, 0.0),          
#     (1.0, 1.0, 0.0),
#     (0.0, 1.0, 0.0),
#     (0.0, 0.0, 0.0)
# ]

class Amrmove(Node):
    #-----------
    # 초기화
    #-----------
    def __init__(self):
        super().__init__('amr_move', namespace='/robot4')
        self.navigator = TurtleBot4Navigator(namespace='/robot4')
        
        self.default_paths = []
        self.mission_paths = []
        self.return_paths = []
        self.mission_targets = []
        self.finished_target = None
        self.path_index = 0
        self.mode = "PATROL"
        self.current_task = False
        self.current_pose = None
        self.path_ranges = []
        self.patrol_waypoints = []

        self.targets = []
        self.pending_targets =[]
        self.known_locations = []
        self.DUPLICATE_THRESHOLD = 0.05
        
        self.sub_wb_xyz = self.create_subscription(String, '/webcam_objects/map_detections', self.xyz_callback, 10)
        self.sub_amcl = self.create_subscription(PoseWithCovarianceStamped, "/robot4/amcl_pose", self.amcl_callback, 10)
        self.pub_xyzr_amr2 = self.create_publisher(String, '/a_to_b', 10)

    #-----------
    # 콜백
    #-----------
    def amcl_callback(self, msg):
        self.current_pose = PoseStamped()
        self.current_pose.header = msg.header
        self.current_pose.pose = msg.pose.pose

    def xyz_callback(self, msg):
        new_targets = self.json_to_dic(msg)
        if not new_targets or len(new_targets) == 0:
            return
        
        added_new = False

        for target in new_targets:
            tx = target["x"]
            ty = target["y"]

            if not self.is_duplicate(tx, ty):
                self.known_locations.append((tx, ty))
                self.pending_targets.append(target)
                added_new = True

        if not added_new:
            return
        if self.mode == "MISSION":
            self.get_logger().info("새좌표 들어옴")
            return
        self.start_pending_mission()

    def start_pending_mission(self):
        if not self.pending_targets:
            return
        
        self.targets = self.pending_targets.copy()
        self.pending_targets.clear()

        self.cancel_task()
        self.sort_xyzr()
        self.path_remake()

    #-----------
    # 데이터 처리
    #-----------
    def is_duplicate(self, target_x, target_y):
        for kx, ky in self.known_locations:
            dist = math.hypot(kx - target_x, ky - target_y)
            if dist <= self.DUPLICATE_THRESHOLD:
                return True
        return False
    
    def json_to_dic(self, msg):
        try:
            data = json.loads(msg.data)

            system_time = data["system_time"]

            targets = []

            OFFSET_X_1 = 0.351
            OFFSET_Y_1 = -0.455

            OFFSET_X_2 = -0.351
            OFFSET_Y_2 = -0.455

            for obj in data["objects"]:
                event_id = obj["event_id"]
                zone = obj["zone"]
                raw_x = obj["x"]
                raw_y = obj["y"]
                if zone == 1 :
                    final_x = raw_x + OFFSET_X_1
                    final_y = raw_y + OFFSET_Y_1
                
                elif zone == 2 :
                    final_x = raw_x + OFFSET_X_2
                    final_y = raw_y + OFFSET_Y_2
                
                else:
                    self.get_logger().warn(f"another zone not save {zone}")
                    continue
                
                targets.append({
                    "event_id": event_id,
                    "zone": zone,
                    "x": final_x,
                    "y": final_y,
                    "time": system_time
                })
                
            return targets
            
        except Exception as e:
            self.get_logger().error(f"json: {e}")
            return []
    
    def publish_target_to_amr2(self, target): 

        data = {
            "event_id": target["event_id"],
            "x": self.current_pose.pose.position.x,
            "y": self.current_pose.pose.position.y,
            "orientation": {
                "x": self.current_pose.pose.orientation.x,
                "y": self.current_pose.pose.orientation.y,
                "z": self.current_pose.pose.orientation.z,
                "w": self.current_pose.pose.orientation.w
            }
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub_xyzr_amr2.publish(msg)
        self.get_logger().info(f"Publish: {data}")
     
    #-----------
    # Utility
    #-----------   
    def make_pose(self, frame_id, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        qw, qx, qy, qz = euler2quat(0.0, 0.0, yaw)

        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose
    
    def nearest_wp(self, pose):
        if not hasattr(self, "patrol_waypoints"):
            return 0

        min_distance = float("inf")
        nearest_index = 0

        for i, wp in enumerate(self.patrol_waypoints):
            dx = pose.pose.position.x - wp.pose.position.x
            dy = pose.pose.position.y - wp.pose.position.y
            distance = dx*dx + dy*dy

            if distance < min_distance:
                min_distance = distance
                nearest_index = i
        
        return nearest_index
    
    def nearest_patrol_path(self):
        if self.current_pose is None:
            return 0
        wp = self.nearest_wp(self.current_pose)

        for i, (start, end) in enumerate(self.path_ranges):
            if start <= wp <= end:
                return i
        
        return 0
    
    #-----------
    # 경로 생성
    #-----------    
    def default_path(self, start):
        if self.current_pose is None:
            self.get_logger().warn("amcl not received")
            return 

        self.default_paths = []
        self.patrol_waypoints = []
        self.path_length = []
        
        targets = PATROL_TARGETS

        self.path_ranges.clear()
        wp_start = 0

        for i in range(len(targets)):
            goal = self.make_pose(
                "map",
                targets[(i+1) % len(targets)][0],
                targets[(i+1) % len(targets)][1],
                targets[(i+1) % len(targets)][2]
            )
            path = self.navigator.getPath(start, goal)

            if path is None:
                self.get_logger().error(f"path{i} no !! ")
                continue

            self.default_paths.append(path)
            self.patrol_waypoints.extend(path.poses)
            wp_end = wp_start + len(path.poses) -1
            self.path_ranges.append((wp_start, wp_end))
            wp_start = wp_end +1
            self.get_logger().info(f"path{i+1}: {len(path.poses)} waypoints")
        
            start = goal
        
        self.path_length.append(0.0)

        for i in range(1, len(self.patrol_waypoints)):
            p1 = self.patrol_waypoints[i - 1].pose.position
            p2 = self.patrol_waypoints[i].pose.position

            d = math.hypot(
                p2.x - p1.x,
                p2.y - p1.y
            )
            self.path_length.append(self.path_length[-1] + d)

        self.get_logger().info(f"Total waypoints: {len(self.patrol_waypoints)}")
        self.get_logger().info(f"total patrol lengrh: {self.path_length[-1]:.2f}")
        self.path_index = 0

    def sort_xyzr(self):
        if self.current_pose is None or not hasattr(self, "path_length") or len(self.path_length) == 0:
            self.get_logger().warn("Patrol path not generated yet. Skiping sort.")
            return
        if self.current_pose is None:
            return 
        
        robot_pose = self.current_pose
        robot_wp = self.nearest_wp(robot_pose)
        robot_s = self.path_length[robot_wp]
        total_length = self.path_length[-1]

        for target in self.targets:
            target_pose = self.make_pose("map", target["x"], target["y"], 0.0)
            target_wp = self.nearest_wp(target_pose)
            target["wp"] = target_wp
            target_s = self.path_length[target_wp]
            distance = target_s - robot_s
            
            if distance < 0:
                distance += total_length
            
            target["distance"] = distance
        
        self.targets.sort(key=lambda t: t["distance"])

    def path_remake(self):
        if self.current_pose is None:
            self.get_logger().warn("amcl not received")
            return
        
        self.mission_paths = []
        self.mission_targets = []

        start = self.current_pose

        for target in self.targets:
            goal = self.make_pose("map", target["x"], target["y"], 0.0)
            path = self.navigator.getPath(start, goal)

            if path is None:
                self.get_logger().warn(f"{target['event_id']} path make fail")
                continue

            self.mission_paths.append(path)
            self.mission_targets.append(target)

            start = goal
        
        if len(self.mission_paths) == 0:
            self.get_logger().warn("all mission fail go to patrol")
            self.return_to_patrol()
            return

        self.path_index = 0
        self.mode = "MISSION"

    def return_to_patrol(self):
        self.return_paths = []
        self.next_patrol_path = self.nearest_patrol_path()

        nearest_wp = self.nearest_wp(self.current_pose)

        start = self.current_pose
        wp_pose = self.patrol_waypoints[nearest_wp]

        connect = self.navigator.getPath(start, wp_pose)
        
        if connect is not None:
            self.return_paths.append(connect)
        
        start_wp, end_wp = self.path_ranges[self.next_patrol_path]
        
        remain_poses = self.patrol_waypoints[nearest_wp+1:end_wp+1]
        if len(remain_poses) > 0:
            remain = Path()
            remain.header = self.default_paths[self.next_patrol_path].header
            remain.poses = remain_poses
            self.return_paths.append(remain)
        if len(self.return_paths) == 0:
            self.get_logger().warn("reutn fail go to next patrol")
            self.mode = "PATROL"
            self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)
            self.known_locations.clear()
            return

        self.mode = "RETURN"
        self.path_index = 0
        self.known_locations.clear()

    #-----------
    # Navigation
    #----------- 
    def cancel_task(self):
        if self.current_task:
            self.navigator.cancelTask()
            
            self.current_task = False 

    def follow_path(self):
        if self.current_pose is None:
            return
        
        if self.mode == "PATROL":
            paths = self.default_paths
        elif self.mode == "MISSION":
            paths = self.mission_paths
        else:
            paths = self.return_paths
        
        if len(paths) == 0:
            return
        
        if not self.current_task:
            self.navigator.followPath(paths[self.path_index])
            self.current_task = True
            self.get_logger().info(f"start path {self.path_index}")
            return
        
        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            self.current_task = False

            if result == TaskResult.SUCCEEDED:
                if self.mode == "MISSION":
                    self.finished_target = self.mission_targets[self.path_index]
                    self.publish_target_to_amr2(self.finished_target)
                    self.get_logger().info(f"target reached: {self.finished_target['event_id']}")
                self.path_index += 1

                if self.mode == "PATROL":
                    if self.path_index >= len(self.default_paths):
                        self.path_index = 0
                
                elif self.mode == "MISSION":
                    if self.path_index >= len(self.mission_paths):
                        if len(self.pending_targets) > 0:
                            self.get_logger().info("mission complete")
                            self.start_pending_mission()

                        else:
                            self.get_logger().info("all clear go to patrol")
                            self.return_to_patrol()
                
                elif self.mode == "RETURN":
                    if self.path_index >= len(self.return_paths):
                        self.mode = "PATROL"
                        self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)

            elif result == TaskResult.FAILED or result == TaskResult.CANCELED:
                self.get_logger().error(f"{self.path_index} fail")
                self.path_index += 1

                if self.mode == "PATROL" and self.path_index >= len(self.default_paths):
                    self.path_index = 0
                elif self.mode == "MISSION" and self.path_index >= len(self.mission_paths):
                    if len(self.pending_targets) > 0:
                        self.get_logger().info("mission failed start pending")
                        self.start_pending_mission()
                    else:
                        self.return_to_patrol()

                elif self.mode == "RETURN" and self.path_index >= len(self.return_paths):
                    self.mode = "PATROL"
                    self.path_index = (self.next_patrol_path + 1) % len(self.default_paths)
#-----------
# main
#-----------    
def main(args=None):
    rclpy.init(args = args)
    node = Amrmove()

    node.get_logger().info("nav 켜지는 중")
    node.navigator.waitUntilNav2Active()
    node.get_logger().info("nav 떴다")

    # initial_pose = node.make_pose("map", INITIAL_X, INITIAL_Y, INITIAL_YAW)
    # node.navigator.setInitialPose(initial_pose)
    # node.get_logger().info("Initial pose send")
    time.sleep(3.0)
    node.get_logger().info("/amcl_pose ... loading")
    while rclpy.ok() and node.current_pose is None:
        rclpy.spin_once(node, timeout_sec=1.0)
    node.get_logger().info("current finish")

    # if node.navigator.getDockedStatus():
    #     node.get_logger().info("robot is docked. undocking...")
    #     node.navigator.undock()
    #     node.get_logger().info("undock complete")
    
    node.default_path(node.current_pose)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
        node.follow_path()
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
