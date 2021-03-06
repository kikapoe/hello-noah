import face_recognition
import cv2
import numpy as np
import glob
import time
from pygame import mixer
import os
import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def getImageWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    names = []
    names_idx = []
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp= np.array(faceImg, 'uint8')
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

Ids,names, faces, names_idx = getImageWithID(path)
recognizer.train(faces, np.array(Ids))
print(recognizer.write('trainingData.yml'))
cv2.destroyAllWindows()

print("face",len(Ids))
bgVDO = cv2.VideoCapture("video/test.mp4")
video_capture = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
print(rec.read('trainingData.yml'))
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# voice_list = [
#     'voice/Hello_Poom.mp3',
#     'voice/Hello_Panisara.mp3',
#     'voice/Hello_Pote.mp3',
# ]

mypath = "assets/NOAH - Voice for TBKK/TBKK - Thai 35 Users"
voice_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Count the number of loop that face not found
# if reach N then system can call the name again
# name_count = [0]*len(known_face_names)


same_face_count = 0
is_same_face = False
is_unknown = False
previous_face = [""]
current_face = [""]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
last_face = []
process_this_frame = True

curr = [] # current face detected
last = [] # previous face detected
start_time = 0
period_time = 1
current_time = 0
sameface = False
sameface_name = "" 
while True: 
    ret, frame = video_capture.read()
    ret2, frame2 = bgVDO.read() 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    curr = []
    # Only process every other frame of video to save time
    if process_this_frame:
        for (x,y,w,h) in faces:
            if(w > 120 and h > 120):
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                id, conf = rec.predict(gray[y:y+h, x:x+w])
                index = Ids.index(id) # order of item in list 
                # index = Ids.index(id) # order of item in list 
                if id in Ids and conf < 80:
                    cv2.putText(frame,str(names[index])+","+str(round(conf,4)),(x,y+h), font, 1,(255,255,255),2,cv2.LINE_AA)
                    curr.append(names[index])
                else :
                    cv2.putText(frame,"Unknown",(x,y+h), font, 1,(255,255,255),2,cv2.LINE_AA)
                    curr.append("unknown")
                pass
        
        print(last)
        print(curr)
        if set(curr).issubset(set(last)) and ( last and curr ) and sameface == False :
            sameface = True
            start_time = time.time()
            sameface_name = curr[0]
            print("True")
        elif not set(curr).issubset(set(last)) and sameface == True:
            start_time = time.time()
            sameface = False
            sameface_name = ""
        elif not curr and not last :
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
            print("say hi :" , sameface_name)
            sameface = False
            sameface_name = ""

        if (time.time() - start_time > period_time) and sameface and sameface_name == "unknown" and sameface_name:
            mixer.init()
            mixer.music.load("voice/Hello.mp3")
            mixer.music.play()
            while mixer.music.get_busy():  # wait for music to finish playing
                time.sleep(1)
            sameface = False
            sameface_name = ""
            
        cv2.imshow("Faces",frame)
        if(cv2.waitKey(1) == ord('q')):
            break
        pass

        last = curr
        print("-----")
    process_this_frame = not process_this_frame

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(0.2)
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
