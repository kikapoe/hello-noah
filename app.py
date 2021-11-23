
from flask import Flask, render_template, Response, redirect,  url_for, jsonify, request
import cv2
import face_recognition
import numpy as np
import time
from pygame import mixer
import logging
import os
import webbrowser
from PIL import Image
from os import listdir
from os.path import isfile, join
from dotenv import load_dotenv
from faker import Faker
#from flask import Flask, Response, jsonify, redirect, request, url_for
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from twilio.twiml.voice_response import Dial, VoiceResponse
from selenium import webdriver

import os
import re


app = Flask(__name__, static_folder='static')
load_dotenv()

app = Flask(__name__)
fake = Faker()
alphanumeric_only = re.compile("[\W_]+")
phone_pattern = re.compile(r"^[\d\+\-\(\) ]+$")

twilio_number = os.environ.get("TWILIO_CALLER_ID")

# Store the most recently created identity in memory for routing calls
IDENTITY = {"identity": ""}

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = os.path.abspath('dataset')
# path = os.path.abspath('C:\\Users\\akkap\\Documents\\Noah\\Face_rec\\dataset')
print(os.path.dirname(os.path.abspath(__file__)))

driver = webdriver.Chrome('http://localhost:5000/')

def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    names = []
    faces = []  # all index of name
    names_idx = []  # unique name
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        NAME = os.path.split(imagePath)[-1].split('.')[0]
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        if NAME not in names:
            names_idx.append(NAME)
        names.append(NAME)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, names, faces, names_idx


Ids, names, faces , names_idx = getImageWithID(path)
recognizer.train(faces, np.array(Ids))
print(recognizer.write('trainingData.yml'))
cv2.destroyAllWindows()

## video_capture = cv2.VideoCapture(0)
bgVDO = cv2.VideoCapture("video/Eye_Emotion_Standby.mp4")

# faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# rec = cv2.face.LBPHFaceRecognizer_create()
# print(rec.read('trainingData.yml'))
# rec.read('trainingData.yml')
# id = 0
# font = cv2.FONT_HERSHEY_SIMPLEX

# voice_list = [
#     'voice/Hello_Poom.mp3',
#     'voice/Hello_Panisara.mp3',
#     'voice/Hello_Pote.mp3',
# ]

# Initialize some variables
curr = []  # current face detected
last = []  # previous face detected
start_time = 0
period_time = 1
current_time = 0
sameface = False
sameface_name = ""
process_this_frame = True


def classify_face():
    video_capture = cv2.VideoCapture(0)
    voice_dir = "./assets/NOAH - Voice for TBKK/TBKK - Thai 35 Users"
    voice_list = [f for f in listdir(voice_dir) if isfile(join(voice_dir, f))]

    # voice_list = [
    #     'voice/Hello_Poom.mp3',
    #     'voice/Hello_Panisara.mp3',
    #     'voice/Hello_Pote.mp3',
    # ]

    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    rec = cv2.face.LBPHFaceRecognizer_create()
    print(rec.read('trainingData.yml'))
    rec.read('trainingData.yml')
    id = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize some variables
    curr = []  # current face detected
    last = []  # previous face detected
    start_time = 0
    period_time = 1
    current_time = 0
    sameface = False
    sameface_name = ""
    process_this_frame = True

    while True:
        ret, frame = video_capture.read()
        ret2, frame2 = bgVDO.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        curr = []
        # Only process every other frame of video to save time
        if process_this_frame:
            for (x, y, w, h) in faces:
                if(w > 120 and h > 120):
                    # print(faces)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    id, conf = rec.predict(gray[y:y+h, x:x+w])
                    index = Ids.index(id)  # order of item in list
                    if id in Ids and conf < 90:
                        curr.append(names[index])
                    else:
                        curr.append("unknown")
                    pass

            app.logger.info(last)
            app.logger.info(curr)
            if set(curr).issubset(set(last)) and (last and curr) and sameface == False:
                sameface = True
                start_time = time.time()
                sameface_name = curr[0]
                print("True")
            elif not set(curr).issubset(set(last)) and sameface == True:
                start_time = time.time()
                sameface = False
                sameface_name = ""
            elif not curr and not last:
                start_time = time.time()
                sameface = False
                sameface_name = ""

            print(time.time() - start_time)
            if (time.time() - start_time > period_time) and sameface and sameface_name != "unknown" and sameface_name:
                index = names_idx.index(sameface_name)
                mixer.init()
                mixer.music.load("assets/NOAH - Voice for TBKK/TBKK - Thai 35 Users/"+voice_list[index])
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
                print("say hi :", sameface_name)
                sameface = False
                sameface_name = ""
                
            
##
            if (time.time() - start_time > period_time) and sameface and sameface_name == "unknown" and sameface_name:
                mixer.init()
                mixer.music.load("voice/Hello.mp3")
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
                sameface = False
                sameface_name = ""
                webbrowser.open(
                    'http://localhost:5000/department')
                    #'https://voice-call-python.herokuapp.com', new=0)
                return  
            # cv2.imshow("Faces", frame)
            # if(cv2.waitKey(1) == ord('q')):
            #     break
            # pass

            last = curr
            print("-----")
        process_this_frame = not process_this_frame

# def screen_server():

#     ret2, frame2 = bgVDO.read()

#     # repeat video mp4 when video is finished
#     if ret2:
#         cv2.imshow('video', frame2)
#     else:
#         bgVDO.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     small_frame = cv2.resize(frame2, (0, 0), fx=0.25, fy=0.25)
#     rgb_small_frame = small_frame[:, :, ::-1]
#     gray = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting image
#     frame = gray.tobytes()
#     yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html',)


@app.route('/department')
def department():
   ## return app.send_static_file("department.html")
    return render_template("department.html")
 # return render_template('department.html',)


@app.route('/recording')
def rec():
    return Response(classify_face(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/ss')
# def ss():
#     return Response(screen_server(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/token", methods=["GET"])
def token():
    # get credentials for environment variables
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    application_sid = os.environ["TWILIO_TWIML_APP_SID"]
    api_key = os.environ["API_KEY"]
    api_secret = os.environ["API_SECRET"]

    # Generate a random user name and store it
    identity = alphanumeric_only.sub("", fake.user_name())
    IDENTITY["identity"] = identity

    # Create access token with credentials
    token = AccessToken(account_sid, api_key, api_secret, identity=identity)

    # Create a Voice grant and add to token
    voice_grant = VoiceGrant(
        outgoing_application_sid=application_sid,
        incoming_allow=True,
    )
    token.add_grant(voice_grant)

    # Return token info as JSON
    token = token.to_jwt()

    # Return token info as JSON
    return jsonify(identity=identity, token=token)


@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    if request.form.get("To") == twilio_number:
        # Receiving an incoming call to our Twilio number
        dial = Dial()
        # Route to the most recently created client based on the identity stored in the session
        dial.client(IDENTITY["identity"])
        resp.append(dial)
    elif request.form.get("To"):
        # Placing an outbound call from the Twilio client
        dial = Dial(caller_id=twilio_number)
        # wrap the phone number or client name in the appropriate TwiML verb
        # by checking if the number given has only digits and format symbols
        if phone_pattern.match(request.form["To"]):
            dial.number(request.form["To"])
        else:
            dial.client(request.form["To"])
        resp.append(dial)
    else:
        resp.say("Thanks for calling!")

    return Response(str(resp), mimetype="text/xml")


if __name__ == '__main__':
    app.run(debug=True)
