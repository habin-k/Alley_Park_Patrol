from django.db import models


class ParkingEvent(models.Model):
    STATUS_CHOICES = [
        ('DETECTED', 'Detected'),
        ('SCANNED', 'Scanned'),
        ('WARNING_ISSUED', 'Warning Issued'),
    ]

    vehicle_id    = models.IntegerField(null=True, blank=True)
    observation_x = models.FloatField()
    observation_y = models.FloatField()
    status        = models.CharField(max_length=15, choices=STATUS_CHOICES,
                                     default='DETECTED', db_index=True)
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'parking_events'


class VehicleInfo(models.Model):
    event        = models.OneToOneField(ParkingEvent, on_delete=models.CASCADE,
                                        related_name='vehicle_info')
    plate_number = models.CharField(max_length=20)
    ocr_image    = models.TextField(null=True, blank=True)  # base64 인코딩된 번호판 이미지
    updated_at   = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'vehicle_info'


class DisabledVehicle(models.Model):
    plate_number  = models.CharField(max_length=20, unique=True)
    registered_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'disabled_vehicle'
