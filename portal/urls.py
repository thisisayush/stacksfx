from django.urls import path
from .views import *

app_name = "portal"

urlpatterns = [
    path("", IndexView, name="home"),
    path("processImages/", FaceDetectionView, name="process-image"),
    path("detect-mood/", MoodDetectionView, name="detect-mood"),
    path("music/<mood>/", GetEmotionSongView, name="get-emotion-music"),
    path("train/", TrainView, name="train"),
]
