from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0010_remove_parkingevent_webcam_image_path'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='parkingevent',
            name='vehicle_x',
        ),
        migrations.RemoveField(
            model_name='parkingevent',
            name='vehicle_y',
        ),
    ]
