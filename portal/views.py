from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from .utils import *
import base64
import time

# Create your views here.


def IndexView(request):

    return render(request, "portal/index.html")

def FaceDetectionView(request):
    
    epoch = str(time.time())
    if request.method != "POST":
        return HttpResponse("Only POST Requests Accepted")

    if "image" in request.POST:
        with open("temp/%s.png"%(epoch), "wb") as fh:
            fh.write(base64.b64decode(request.POST.get("image")))

    f = FaceRecogniser()
    if not f:
        return HttpResponse("Error!")
    
    face = f.detect_face(f.process_image("temp/%s.png"%epoch))

    if face == -1 or face == -2:
        return JsonResponse({"status": False, "msg": "No/Multiple Face Datected!"})
    
    return JsonResponse({"status": True})
