import face_recognition
import cv2
import numpy as np
import glob
import time
from pygame import mixer

bgVDO = cv2.VideoCapture("video/test.mp4")
video_capture = cv2.VideoCapture(0)

poom_image_list = []
praew_image_list = []
pote_image_list = []

poom_image_list.append(face_recognition.load_image_file('poom/poom_1.jpg'))
praew_image_list.append(face_recognition.load_image_file('praew/praew_1.jpg'))
pote_image_list.append(face_recognition.load_image_file('pote/pote_1.jpg'))

poom_face_encoding = face_recognition.face_encodings(poom_image_list[0])[0]
praew_face_encoding = face_recognition.face_encodings(praew_image_list[0])[0]
pote_face_encoding = face_recognition.face_encodings(pote_image_list[0])[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    poom_face_encoding,
    praew_face_encoding,
    pote_face_encoding,
]

known_face_names = [
    "Poom",
    "Praew",
    "Pote"
]

voice_list = [
    'voice/Hello_Poom.mp3',
    'voice/Hello_Panisara.mp3',
    'voice/Hello_Pote.mp3',
]

# Count the number of loop that face not found
# if reach N then system can call the name again
name_count = [0]*len(known_face_names)


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

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    ret2, frame2 = bgVDO.read()

    # repeat video mp4 when video is finished
    if ret2:
        cv2.imshow('video', frame2)
    else:
        bgVDO.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # small_frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # rgb_small_frame2 = small_frame2[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        current_face = []
        provious_face = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            current_face.append(name)
            
            if name != "Unknown" and name not in last_face:
                last_face.append(name)
                
            
            if name == "Unknown" and same_face_count > 5 :
                mixer.init()
                mixer.music.load(voice_list[known_face_names.index(item)])
                mixer.music.play()
                while mixer.music.get_busy():  # wait for music to finish playing
                    time.sleep(1)
                same_face_count = 0
            
        is_same_face = not set(current_face).issubset(set(previous_face))
        
        print(is_same_face)
        if not is_same_face:
            same_face_count = 0
        else:
            same_face_count += 1

        previous_face = current_face
        current_face = current_face.clear()

        print(same_face_count)

        # If detect then trigger voice
        # If not detect after N loop -> trigger voice again
        for item in last_face:
            if item not in face_names:
                # run number until 60 then reset
                if name_count[known_face_names.index(item)] <= 60:
                    name_count[known_face_names.index(item)] += 1
                else:
                    name_count[known_face_names.index(item)] = 0
                    last_face.remove(item)
            else:
                if name_count[known_face_names.index(item)] == 0 and same_face_count > 3:
                    mixer.init()
                    mixer.music.load(voice_list[known_face_names.index(item)])
                    mixer.music.play()
                    while mixer.music.get_busy():  # wait for music to finish playing
                        time.sleep(1)
                    same_face_count = 0
                    name_count[known_face_names.index(item)] += 1
                
            print(last_face)
            print(name_count)
            print("-----")

    process_this_frame = not process_this_frame

    # Display the results
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/4 size
    #     top *= 4
    #     right *= 4
    #     bottom *= 4
    #     left *= 4

    #     # Draw a box around the face
    #     cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv2.rectangle(frame, (left, bottom - 35),
    #                   (right, bottom), (0, 0, 255), cv2.FILLED)
    #     font = cv2.FONT_HERSHEY_DUPLEX
    #     cv2.putText(frame, name, (left + 6, bottom - 6),
    #                 font, 1.0, (255, 255, 255), 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting image
    # cv2.imshow('Video', gray)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
