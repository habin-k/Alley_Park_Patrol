from rest_framework import serializers
from .models import ParkingEvent, VehicleInfo, DisabledVehicle


class ParkingEventCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingEvent
        fields = ['observation_x', 'observation_y']


class ParkingEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingEvent
        fields = '__all__'


class ParkingEventNextSerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingEvent
        fields = ['id', 'zone_type', 'observation_x', 'observation_y', 'status', 'created_at']



class ZoneUpdateSerializer(serializers.Serializer):
    vehicle_type = serializers.ChoiceField(choices=['NORMAL', 'ILLEGAL'])
    zone_type    = serializers.ChoiceField(choices=['Not', 'NORMAL', 'DISABLED', 'FIRE'])


class VehicleInfoCreateSerializer(serializers.Serializer):
    event_id       = serializers.IntegerField()
    plate_number   = serializers.CharField(max_length=20)
    amr_vehicle_x  = serializers.FloatField(required=False, allow_null=True)
    amr_vehicle_y  = serializers.FloatField(required=False, allow_null=True)
    ocr_image_path = serializers.CharField(required=False, allow_null=True, allow_blank=True)


class VehicleInfoNextSerializer(serializers.ModelSerializer):
    class Meta:
        model = VehicleInfo
        fields = ['id', 'event_id', 'plate_number', 'amr_vehicle_x', 'amr_vehicle_y']


class DisabledVehicleSerializer(serializers.ModelSerializer):
    class Meta:
        model = DisabledVehicle
        fields = ['plate_number', 'registered_at']
