import face_recognition
import cv2
import numpy as np
import csv
import os
import glob
from datetime import datetime

video=cv2.VideoCapture(0)  #0 because for default camera is used
modi_pic=face_recognition.load_image_file("photos/modi.png")
modi_encoding=face_recognition.face_encodings(modi_pic)[0]

rishi_pic=face_recognition.load_image_file("photos/rishi.jpg")
rishi_encoding=face_recognition.face_encodings(rishi_pic)[0]

eknath_pic=face_recognition.load_image_file("photos/eknath.jpg")
eknath_encoding=face_recognition.face_encodings(eknath_pic)[0]

dada_pic=face_recognition.load_image_file("photos/dada.jpg")
dada_encoding=face_recognition.face_encodings(dada_pic)[0]

putin_pic=face_recognition.load_image_file("photos/putin.jpg")
putin_encoding=face_recognition.face_encodings(putin_pic)[0]

known_face_encoding=[
    modi_encoding,
    rishi_encoding,
    eknath_encoding,
    dada_encoding,
    putin_encoding
]

known_face_names=[
    "Narendra Modi",
    "Rishi Sunak",
    "Eknath Shinde",
    "Ajit Pawar",
    "Putin"
]

students=known_face_names.copy()

face_locations=[]
face_encondings=[]
face_names=[]
s=True

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")

file=open(current_date+'.csv','w+',newline='')
lnwriter=csv.writer(file)

while True:
    _,frame=video.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encondings=face_recognition.face_encodings(rgb_small_frame)
        face_names=[]
        for face_encoding in face_encondings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance=face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_face_names[best_match_index]
            
            face_names.append(name)

            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime("%H-%M-%S")
                    lnwriter.writerow(name,current_time)
    
    cv2.imshow("Attendance System",frame)
    if cv2.waitKey(1) & 0xFF==ord('e'):
        break

video.release()
cv2.destroyAllWindows()
file.close()


