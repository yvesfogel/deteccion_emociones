### General imports ###
from __future__ import division

import numpy as np
from numpy.linalg import norm
import pandas as pd
import cv2

from time import time
from time import sleep
import re
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import argparse
from collections import OrderedDict

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage

import dlib

import keras

from tensorflow.keras.models import load_model
from imutils import face_utils

import requests

global shape_x
global shape_y
global input_shape
global nClasses

from tkinter import filedialog
from tkinter import *



def show_webcam(vid, save) :

    shape_x = 48
    shape_y = 48
    input_shape = (shape_x, shape_y, 1)
    nClasses = 7

    thresh = 0.25
    frame_check = 20


    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_face(frame):
        
        cascPath = 'Models/face_landmarks.dat'
        faceCascade = cv2.CascadeClassifier(cascPath)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6,
                                                      minSize=(shape_x, shape_y),
                                                      flags=cv2.CASCADE_SCALE_IMAGE)
        coord = []
                                                      
        for x, y, w, h in detected_faces :
            if w > 100 :
                sub_img=frame[y:y+h,x:x+w]
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255,255),1)
                coord.append([x,y,w,h])

        return gray, detected_faces, coord

    def extract_face_features(faces, offset_coefficients=(0.075, 0.05)):
        gray = faces[0]
        detected_face = faces[1]
        
        new_face = []
        
        for det in detected_face :
            x, y, w, h = det
            
            horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
            vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
            
            extracted_face = gray[y+vertical_offset:y+h, x+horizontal_offset:x-horizontal_offset+w]
            
            new_extracted_face = zoom(extracted_face, (shape_x / extracted_face.shape[0],shape_y / extracted_face.shape[1]))
            new_extracted_face = new_extracted_face.astype(np.float32)
            new_extracted_face /= float(new_extracted_face.max())
            
            new_face.append(new_extracted_face)
        
        return new_face


    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

    (eblStart, eblEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (ebrStart, ebrEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]

    model = load_model('../Models/video.h5')
    face_detect = dlib.get_frontal_face_detector()
    predictor_landmarks  = dlib.shape_predictor("../Models/face_landmarks.dat")
    
    video_capture = cv2.VideoCapture(vid)
    
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    out = cv2.VideoWriter(save, fourcc, 20.0, (1920, 1080), 1)

    frames_number = 0

    while True:
        ret, frame = video_capture.read()


        if (ret == False):
            break

        frames_number += 1

        brillo = brightness(frame)


        
        face_index = 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)

        for (i, rect) in enumerate(rects):

            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            face = gray[y:y+h,x:x+w]
            
            try:
                face = zoom(face, (shape_x / face.shape[0],shape_y / face.shape[1]))
            except:
                continue
            
            face = face.astype(np.float32)
            
            face /= float(face.max())
            face = np.reshape(face.flatten(), (1, 48, 48, 1))

            
            prediction = model.predict(face)
            prediction_result = np.argmax(prediction)

            current_frame = video_capture.get(1)


            color = (0, 255, 0)


            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
            cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
     
            for (j, k) in shape:
                cv2.circle(frame, (j, k), 1, (0, 0, 255), -1)
            
            cv2.putText(frame, "----------------",(40,100 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Emotional report : Face #" + str(i+1),(40,120 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Angry : " + str(round(prediction[0][0],3)),(40,140 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Disgust : " + str(round(prediction[0][1],3)),(40,160 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 0)
            cv2.putText(frame, "Fear : " + str(round(prediction[0][2],3)),(40,180 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Happy : " + str(round(prediction[0][3],3)),(40,200 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Sad : " + str(round(prediction[0][4],3)),(40,220 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Surprise : " + str(round(prediction[0][5],3)),(40,240 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)
            cv2.putText(frame, "Neutral : " + str(round(prediction[0][6],3)),(40,260 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)


            cv2.putText(frame, "Nivel de brillo : {:.1f}".format(brillo),(40,400 + 180*i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155, 1)



            if prediction_result == 0 :
                cv2.putText(frame, "Angry",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif prediction_result == 1 :
                cv2.putText(frame, "Disgust",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif prediction_result == 2 :
                cv2.putText(frame, "Fear",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif prediction_result == 3 :
                cv2.putText(frame, "Happy",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif prediction_result == 4 :
                cv2.putText(frame, "Sad",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            elif prediction_result == 5 :
                cv2.putText(frame, "Surprise",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            else :
                cv2.putText(frame, "Neutral",(x+w-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # 3. Eye Detection and Blink Count
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            
            # Compute Eye Aspect Ratio
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            
            # And plot its contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            # 4. Detect Nose
            nose = shape[nStart:nEnd]
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)

            # 5. Detect Mouth
            mouth = shape[mStart:mEnd]
            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # 6. Detect Jaw
            jaw = shape[jStart:jEnd]
            jawHull = cv2.convexHull(jaw)
            cv2.drawContours(frame, [jawHull], -1, (0, 255, 0), 1)
            
            # 7. Detect Eyebrows
            ebr = shape[ebrStart:ebrEnd]
            ebrHull = cv2.convexHull(ebr)
            cv2.drawContours(frame, [ebrHull], -1, (0, 255, 0), 1)
            ebl = shape[eblStart:eblEnd]
            eblHull = cv2.convexHull(ebl)
            cv2.drawContours(frame, [eblHull], -1, (0, 255, 0), 1)
        
        cv2.putText(frame,'Number of Faces : ' + str(len(rects)),(40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 155, 1)

        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('Video', frame)


    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

def brightness(img):
    try:
        if len(img.shape) == 3:
            return np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            return np.average(img)
    except:
        pass


def main(vid, save):
    show_webcam(vid, save)

if __name__ == "__main__":

    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
    root.savename =  filedialog.asksaveasfile(filetypes = [("mp4 files","*.mp4")])

    main(root.filename, root.savename.name)
