import cv2
import numpy as np



cap = cv2.VideoCapture(0)
# cv2.createTrackbar('Threshold', '')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

while(True):

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    


    for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if(len(eyes)):
                leftEye = eyes[0]
                ex = leftEye[0]
                ey = leftEye[1]
                ew = leftEye[2]
                eh = leftEye[3]
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            # v2.putText(roi_color, 'A', (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    # print('Inicio')
    # print(eyes)
    # print('fim')

    cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    k = cv2.waitKey(20) & 0xFF
    if(k == 27):
        break

cv2.destroyAllWindows()
cap.release()
