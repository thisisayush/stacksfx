from django.urls import path
from .views import *

app_name = "portal"

urlpatterns = [
    path("", IndexView, name="home"),
    path("processImages/", FaceDetectionView, name="process-image"),
]