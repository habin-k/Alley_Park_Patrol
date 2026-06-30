from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'final_pjt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dg',
    maintainer_email='donggeun3237@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'amr1_goal_publisher = final_pjt.amr1_goal_publisher:main',
            'amr2_mission = final_pjt.amr2_mission:main',
            'amr2_mission_lidar = final_pjt.amr2_mission_lidar:main',
            'amr2_mission_yolo = final_pjt.amr2_mission_yolo:main',
            'amr2_mission_yolo_depth = final_pjt.amr2_mission_yolo_depth:main',
            'fake_node = final_pjt.fake_node:main',
            'fake_ocr_node = final_pjt.fake_ocr_node:main',
            'plate_ocr_node = final_pjt.plate_ocr_node:main',
            'plate_lida_run_test = final_pjt.plate_lida_run_test:main',
        ],
    },
)
