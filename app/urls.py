from django.urls import path
from .monitor_dashboard import dashboard, login_view, logout_view
from .api_views import (
    parking_create, parking_list,
    parking_delete, parking_by_vehicle,
    vehicle_create,
    vehicle_get, vehicle_verify, disabled_check,
)
from .monitor_views import (
    api_login, api_logout,
    summary, event_list, disabled_list, disabled_register,
    disabled_delete,
    webcam_frame_upload, webcam_frame_get, webcam_image, webcam_stream,
    webcam2_frame_upload, webcam2_frame_get, webcam2_stream,
    amr1_frame_upload, amr1_frame_get,
    amr2_frame_upload, amr2_frame_get,
)

urlpatterns = [
    # 인증 (HTML)
    path('login/', login_view, name='login'),
    path('logout/', logout_view, name='logout'),

    path('monitor/', dashboard, name='monitor_dashboard'),

    # 인증 API (Swagger)
    path('api/login/', api_login, name='api_login'),
    path('api/logout/', api_logout, name='api_logout'),

    # API - 웹캠 노드
    path('api/parking/', parking_create, name='api_parking_create'),
    path('api/parking/list/', parking_list, name='api_parking_list'),
    path('api/parking/<int:event_id>/delete/', parking_delete, name='api_parking_delete'),

    # API - AMR1 (Police 1)
    path('api/vehicle/', vehicle_create, name='api_vehicle_create'),
    path('api/disabled/<str:plate_number>/', disabled_check, name='api_disabled_check'),

    # API - OCR 브리지용
    path('api/parking/by-vehicle/<int:vehicle_id>/', parking_by_vehicle, name='api_parking_by_vehicle'),
    path('api/vehicle/<int:event_id>/', vehicle_get, name='api_vehicle_get'),

    # API - Police 2
    path('api/vehicle/verify/', vehicle_verify, name='api_vehicle_verify'),

    # API - 모니터링
    path('api/monitor/summary/', summary, name='api_monitor_summary'),
    path('api/monitor/events/', event_list, name='api_monitor_events'),
    path('api/monitor/disabled/', disabled_list, name='api_monitor_disabled_list'),
    path('api/monitor/disabled/register/', disabled_register, name='api_monitor_disabled_register'),
    path('api/monitor/disabled/<str:plate_number>/', disabled_delete, name='api_monitor_disabled_delete'),
    path('api/webcam1/frame/', webcam_frame_upload, name='api_webcam_upload'),
    path('api/webcam1/frame/latest/', webcam_frame_get, name='api_webcam_get'),
    path('api/webcam1/image/', webcam_image, name='api_webcam_image'),
    path('api/webcam1/stream/', webcam_stream, name='api_webcam_stream'),
    path('api/webcam2/frame/', webcam2_frame_upload, name='api_webcam2_upload'),
    path('api/webcam2/frame/latest/', webcam2_frame_get, name='api_webcam2_get'),
    path('api/webcam2/stream/', webcam2_stream, name='api_webcam2_stream'),
    path('api/amr1/frame/', amr1_frame_upload, name='api_amr1_upload'),
    path('api/amr1/frame/latest/', amr1_frame_get, name='api_amr1_get'),
    path('api/amr2/frame/', amr2_frame_upload, name='api_amr2_upload'),
    path('api/amr2/frame/latest/', amr2_frame_get, name='api_amr2_get'),
]
