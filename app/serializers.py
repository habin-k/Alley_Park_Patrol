from rest_framework import serializers
from .models import ParkingEvent, VehicleInfo, DisabledVehicle


class ParkingEventCreateSerializer(serializers.Serializer):
    event_id      = serializers.IntegerField(required=False, allow_null=True)
    observation_x = serializers.FloatField()
    observation_y = serializers.FloatField()


class ParkingEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingEvent
        fields = '__all__'



class VehicleInfoCreateSerializer(serializers.Serializer):
    event_id     = serializers.IntegerField()
    plate_number = serializers.CharField(max_length=20)
    ocr_image    = serializers.CharField(required=False, allow_null=True, allow_blank=True)  # base64



class DisabledVehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = DisabledVehicle
        fields = ['plate_number', 'registered_at']
