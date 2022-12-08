from ctypes.wintypes import RGB
import requests
import numpy as np
import cv2
import time

# Don't forget to change ur own token! 


url = 'https://notify-api.line.me/api/notify'
token = 'changed ur token here'
headers = {
            'content-type':
            'application/x-www-form-urlencoded',
            'Authorization':'Bearer '+token
           }

set_id_checker = set()
face_cascade = cv2.CascadeClassifier('lib_file/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("lib_file/trainingdata.yml")
id=0
font= cv2.FONT_HERSHEY_COMPLEX_SMALL,12,2,0,4
def greeting(id):
    #print("Welcome :",id)
    msg = f"User : {id} Just Checking in!"
    msg_2 = f"Counting {len(set_id_checker)} People Checked in!"
    r = requests.post(url, headers=headers , data = {'message':msg})
    r = requests.post(url, headers=headers , data = {'message':msg_2})
    print(r.text)
    return id
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="Yo"
        if id==2:
            id="Graphic"
        if id==3:
            id="GoKingXD"
        if id==4:
            id="Tor"
        if id==5:
            id='Bell'
        if id==6:
            id="Riw"
        if id not in set_id_checker:
            set_id_checker.add(id)
            greeting(id)
        cv2.putText(img, str(id), (x, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('img',img)
    #print("Welcome :",id)
    if cv2.waitKey(1) == ord('y'):
        break
    elif cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()