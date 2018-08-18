from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from .utils import *
import base64
import time
import os
import shutil
from django.contrib.staticfiles.templatetags.staticfiles import static
# Create your views  here.


def IndexView(request):
    is_first_run = True
    if "is_first_run" in request.session and request.session['is_first_run'] != True:
        is_first_run = False
        if "sessionid" not in request.session:
            request.session['sessionid'] = str(time.time())   
    else:
        request.session['sessionid'] = str(time.time())
    
    if not os.path.exists("temp/%s"%request.session['sessionid']):
        os.makedirs("temp/%s"%request.session['sessionid'])

    return render(request, "portal/index.html", {"is_first_run": is_first_run})

def FaceDetectionView(request):
    
    epoch = str(time.time())
    if request.method != "POST":
        return HttpResponse("Only POST Requests Accepted")

    file_name = "temp/%s.png" % epoch
    if "image" in request.POST:
        with open(file_name, "wb") as fh:
            img = request.POST.get('image')
            
            missing_padding = len(img) % 4
            if missing_padding != 0:
                img += '='* (4 - missing_padding)
            
            fh.write(base64.b64decode(img))

    f = FaceRecogniser(request.session['sessionid'])
    if not f:
        return HttpResponse("Error!")
    
    face = f.detect_face(file_name)

    if os.path.exists(file_name):
        os.remove(file_name)
    
    if type(face) == int and (face == -1 or face == -2):
        return JsonResponse({"status": False, "msg": "No/Multiple Face Datected!"})
    
    return JsonResponse({"status": True})

def MoodDetectionView(request):

    images = []
    if request.method != "POST":
        return JsonResponse({"success": False, "msg": "Only POST Requests are accepted"})
    
    img_len = 16
    d = "temp/"+str(time.time())
    if not os.path.exists(d):
        os.makedirs(d)

    for i in range(img_len):
        img = request.POST.get("image"+str(i), None)
        
        if img is not None:
            missing_padding = len(img) % 4
            if missing_padding != 0:
                img += '='* (4 - missing_padding)

            file_name = "%s/%s.png"%(d,str(time.time()))

            with open(file_name, "wb") as fh:
                fh.write(base64.b64decode(img))
                images.append(file_name)
    
    f = FaceRecogniser(request.session['sessionid'])
    to_return = {}
    mood = f.run_detection(images)
    to_return = {"success": True, "mood": mood}
    
    if os.path.exists(d):
        shutil.rmtree(d)
    
    return JsonResponse(to_return)


def GetEmotionSongView(request, mood):

    s = SongPredictor()

    return JsonResponse(s.choose_random_action(mood))

def TrainView(request):

    images = {
        "happy": [],
        "sad": []
    }
    if request.method != "POST":
        return JsonResponse({"success": False, "msg": "Only POST Requests are accepted"})
    
    img_len = 16
    d = "temp/train-"+str(time.time())
    if not os.path.exists(d):
        os.makedirs(d)

    f = FaceRecogniser(request.session['sessionid'])
    
    for mood in f.emotions:
        for i in range(img_len):
            img = request.POST.get("%s-image%s"%(mood, str(i)), None)
            
            if img is not None:
                missing_padding = len(img) % 4
                if missing_padding != 0:
                    img += '='* (4 - missing_padding)

                file_name = "%s/%s.png"%(d,str(time.time()))

                with open(file_name, "wb") as fh:
                    fh.write(base64.b64decode(img))
                    images[mood].append(file_name)
    
    to_return = {}
    
    accuracy = f.update_model(images)
    to_return = {"success": True, "accuracy": accuracy}
    request.session['is_first_run'] = False

    if os.path.exists(d):
        shutil.rmtree(d)
    
    return JsonResponse(to_return)