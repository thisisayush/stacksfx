import cv2
import numpy as np
import argparse
import time
import glob
import os
import sys
import subprocess
import pandas
# import random
# import Update_Model
import math
import base64
import spotipy
import spotipy.oauth2 as oauth2
import requests
import json
import random

class FaceRecogniser:
        
    #load classifier
    facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    fishface = cv2.face.FisherFaceRecognizer_create()

    # Initialize Variables
    facedict = {}

    # Define Emotions
    emotions = ["happy", "sad"]


    def __init__(self, session, training_mode=False):
        self.session = session
        try:
            print("Checking if model is trained...")
            self.fishface.read("temp/%s/trained_emoclassifier.xml"%self.session)
            print("Models are trained. We're good to go...")
        except Exception as e:
            if training_mode:
                print("Training Mode is on!")
            if not training_mode:
                print("Error: You do not have a trained model, please run program with --update flag first")

    
    def open_stuff(self, filename): 
        """Open the file using native system player"""
        print("Playing: "+filename)
        
        filename = "music/%s" % (filename)
        
        if sys.platform == "win32":
            # For Windows
            os.startfile(filename)
        else:
            # For UNIX
            opener ="open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    def update_model(self, images):
        """Updates Model and trains it with new data"""

        print("Model update mode active")
        
        # Check for existence of dataset folders
        self.check_folders()
        
        # Get Training Data for each emotion
        for i in range(0, len(self.emotions)):
            self.save_face(self.emotions[i], images[self.emotions[i]])

        print("collected images, looking good! Now updating model...")
        u = update(self.session, self.emotions)
        print("Done!")
        return u.accuracy

    def save_face(self, emotion, images):
        """Save face data for emotion"""

        print("\n\nSaving faces for emotion %s with %s images" % (emotion, str(len(images))))
        self.facedict = {}
        for image in images:
            self.detect_face(image)
        
        # Store faces in dataset directory
        for x in self.facedict.keys():
            print("Creating Image for emotion %s in dataset/%s/"%(emotion,emotion))
            cv2.imwrite("temp/%s/dataset/%s/%s.jpg" %(self.session, emotion, len(glob.glob("temp/%s/dataset/%s/*" %(self.session,emotion)))), self.facedict[x])
        
        # # Empty the dict
        
    
    def detect_face(self, image):
        """Detects and returns a face from a frame"""
        clahe_image = self.process_image(image)
        face = self.facecascade.detectMultiScale(
                clahe_image, 
                scaleFactor=1.1, 
                minNeighbors=15, 
                minSize=(10, 10), 
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        if len(face) == 1: 
            faceslice = self.crop_face(clahe_image, face)
            # cv2.imshow("detect", faceslice) 
            return faceslice
        else:
            if len(face) == 0:
                print("\r Error: No Face Detected!")
                return -1
            else:
                print("\r Error: Multiple Faces Detected!")
                return -2

    def crop_face(self, clahe_image, face):
        """Clear out face from frame"""
        for (x, y, w, h) in face:
            faceslice = clahe_image[y:y+h, x:x+w]
            faceslice = cv2.resize(faceslice, (350, 350))
        print("cropping face")
        self.facedict["face%s" %(len(self.facedict)+1)] = faceslice
        return faceslice

    
    def check_folders(self):
        """Checks for dataset folders for training"""
        
        for x in self.emotions:
            if os.path.exists("temp/%s/dataset/%s" %(self.session, x)):
                pass
            else:
                os.makedirs("temp/%s/dataset/%s" %(self.session, x))

    def recognize_emotion(self):
        """Recognizes the emotion, selects a random file on the emotion and plays it"""

        predictions = []
        confidence = []
        
        for x in self.facedict.keys():
            pred, conf = self.fishface.predict(self.facedict[x])
            cv2.imwrite("images/%s.jpg" %x, self.facedict[x])
            predictions.append(pred)
            confidence.append(conf)
        
        # print(max(set(predictions), key=predictions.count))
        ind = max(set(predictions), key=predictions.count)
        if ind < 0 or ind > len(self.emotions):
            ind = random.randint(0,len(self.emotions)-1)
            
        recognized_emotion = self.emotions[ind]
        
        print("I think you're %s" %recognized_emotion)
        return recognized_emotion
        
    
    def process_image(self, image):
        """Captures and returns a single frame from webcam"""
        # ret, frame = self.video_capture.read()
        frame = cv2.imread(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        return clahe_image

    def run_detection(self, images):
        """Starts the detection"""
        print("Detecting Face")

        for image in images:
            self.detect_face(image)
        
        return self.recognize_emotion()

class update:

    # emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
    fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
    emotions = []
    data = {}

    def get_files(self, emotion): #Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("temp/%s/dataset/%s/*" %(self.session, emotion))
        random.shuffle(files)
        training = files[:int(len(files)*0.8)] #get first 80% of file list
        prediction = files[-int(len(files)*0.2):] #get last 20% of file list
        return training, prediction

    def make_sets(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in self.emotions:
            training, prediction = self.get_files(emotion)
            #Append data to training and prediction list, and generate labels 0-7
            for item in training:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                training_data.append(gray) #append image array to training data list
                training_labels.append(self.emotions.index(emotion))
        
            for item in prediction: #repeat above process for prediction set
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                prediction_data.append(gray)
                prediction_labels.append(self.emotions.index(emotion))

        return training_data, training_labels, prediction_data, prediction_labels

    def run_recognizer(self):
        training_data, training_labels, prediction_data, prediction_labels = self.make_sets()
        
        print("training fisher face classifier")
        print("size of training set is:", len(training_labels), "images")
        self.fishface.train(training_data, np.asarray(training_labels))
        
        print("predicting classification set")
        cnt = 0
        correct = 0
        incorrect = 0
        for image in prediction_data:
            pred, conf = self.fishface.predict(image)
            if pred == prediction_labels[cnt]:
                correct += 1
                cnt += 1
            else:
                incorrect += 1
                cnt += 1
        accuracy = ((100*correct)/(correct + incorrect))
        print("Accuracy: " + str(accuracy))
        return accuracy

    def __init__(self, session, ems):
        self.session = session
        try:
            self.fishface.read("temp/%s/trained_emoclassifier.xml"%self.session)
        except:
            print("Training Initial Emotions")
            self.emotions = ["happy", "sad"]
            self.run_recognizer()
        
        print("Updating Model")
        self.emotions = ems
        self.accuracy = self.run_recognizer()
        self.fishface.write("temp/%s/trained_emoclassifier.xml"%self.session)
    
    def __del__(self):
        self.fishface.write("temp/%s/trained_emoclassifier.xml"%self.session)
    # #Now run it
    # metascore = []
    # for i in range(0,10):
    #     correct = run_recognizer()
    #     print("got", correct, "percent correct!")
    #     metascore.append(correct)

    # print("\n\nend score:", np.mean(metascore), "percent correct!")


class SongPredictor:

    actions = {}

    genre_mapping = {
        "happy": ["pop", "happy"],
        "sad": ["sad",  "afrobeat"]
    }

    # def __init__(self):
    #     self.populate_from_spotify()
    #     # self.readExcelMapping()

    # def readExcelMapping(self):

    #     # Read Emotion-Song Mapping from Excel
    #     df = pandas.read_excel("EmotionLinks.xlsx") #open Excel file
    #     #self.actions["angry"] = [x for x in df.angry.dropna()] #We need de dropna() when columns are uneven in length, which creates NaN values at missing places. The OS won't know what to do with these if we try to open them.
    #     self.actions["happy"] = [x for x in df.happy.dropna()]
    #     self.actions["sad"] = [x for x in df.sad.dropna()]
    #     #self.actions["neutral"] = [x for x in df.neutral.dropna()]


    def predict_genre(self, emotion):
        
        ind = random.randint(0, len(self.genre_mapping[emotion])-1)

        return self.genre_mapping[emotion][ind]

    def choose_random_action(self, emotion):
        genre = self.predict_genre(emotion)
        ind = 0
        while len(self.actions.get(genre, [])) <= 0:
            genre = self.predict_genre(emotion)
            self.populate_from_spotify(genre)
            # get list of files for detected emotion
            # actionlist = [x for x in self.actions[genre]] 
            try: 
                ind = random.randint(0, len(self.actions[genre])-1)
                if ind > len(self.actions[genre]):
                    ind = len(self.actions[genre])
            except ValueError:
                ind = 0
            # Randomly shuffle the list
            # self.open_stuff(actionlist[0])
        return self.actions[genre][ind]

    def populate_from_spotify(self, genre):
        print("Contacting Spotify")
        headers = {
            "Authorization": "Bearer %s"%generate_spotify_token()
        }

        r = requests.get("https://api.spotify.com/v1/recommendations/available-genre-seeds",
                headers=headers)
        if r.status_code == 200:    
            data = json.loads(r.text)
        self.actions[genre] = []
        q = requests.get("https://api.spotify.com/v1/search?q=genre:" + genre + "&type=track", headers=headers)
        
        dataq = json.loads(q.text)
        count = 0
        for track in dataq['tracks']['items']:
            print("Processing Track %s" % str(count))
            obj = {
                "id": track['id'],
                "name": track['name'],
                'genre': genre,
                'preview_url': track['preview_url'],
                'albumart': track['album']['images'][0],
                "url": track['external_urls']['spotify']
            }
            self.actions[genre].append(obj)
            if count == 20:
                break
        print("Spotify End")
    
def generate_spotify_token():
    credentials = oauth2.SpotifyClientCredentials(
        client_id='4fe3fecfe5334023a1472516cc99d805',
        client_secret='0f02b7c483c04257984695007a4a8d5c')
    token = credentials.get_access_token()

    return token
