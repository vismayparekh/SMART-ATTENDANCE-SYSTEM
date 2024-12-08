from django.db import models

class RegisteredUser(models.Model):
    name = models.CharField(max_length=100)
    face_embedding = models.TextField()  # Store embedding as JSON string
    image = models.ImageField(upload_to='registered_faces/')
    bounding_box = models.JSONField(null=True, blank=True)  # Optional for storing YOLO bounding box

    def __str__(self):
        return self.name
