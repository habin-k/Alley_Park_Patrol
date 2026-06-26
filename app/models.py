from django.db import models


ZONE_CHOICES = [
    ('Not', '주차구역 아님'),
    ('NORMAL', '정상 주차구역 (주황색)'),
    ('DISABLED', '장애인 전용'),
    ('FIRE', '소방차 전용'),
]



class ParkingEvent(models.Model):
    VEHICLE_CHOICES = [
        ('NORMAL', 'Normal'),
        ('ILLEGAL', 'Illegal'),
    ]
    STATUS_CHOICES = [
        ('DETECTED', 'Detected'),
        ('SCANNED', 'Scanned'),
        ('WARNING_ISSUED', 'Warning Issued'),
    ]

    vehicle_type      = models.CharField(max_length=10, choices=VEHICLE_CHOICES, null=True, blank=True)
    zone_type         = models.CharField(max_length=10, choices=ZONE_CHOICES, null=True, blank=True)
    observation_x     = models.FloatField()
    observation_y     = models.FloatField()
    status            = models.CharField(max_length=15, choices=STATUS_CHOICES,
                                         default='DETECTED', db_index=True)
    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'parking_events'


class VehicleInfo(models.Model):
    event          = models.OneToOneField(ParkingEvent, on_delete=models.CASCADE,
                                          related_name='vehicle_info')
    plate_number   = models.CharField(max_length=20)
    ocr_image_path = models.TextField(null=True, blank=True)
    amr_vehicle_x  = models.FloatField(null=True, blank=True)
    amr_vehicle_y  = models.FloatField(null=True, blank=True)
    updated_at     = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'vehicle_info'



class DisabledVehicle(models.Model):
    plate_number  = models.CharField(max_length=20, unique=True)
    registered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'disabled_vehicle'
