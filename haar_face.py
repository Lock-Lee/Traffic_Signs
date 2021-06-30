import numpy as np
import cv2
import requests

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# cap = cv2.VideoCapture(1)
# if using IPcam
url = 'http://192.168.1.7/SnapshotJPEG?Resolution=640x480'
# i=0
# import time

while 1:
    # ret, img = cap.read()
    img_resp=requests.get(url)
    img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
    # s=time.time()
    img = cv2.imdecode(img_arr,1)
    # print('imgggg',img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('grayyy',gray)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    add=[]
    # print('type of faces',type(faces),'\n',faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        # cv2.imshow('Haarcascades',roi_color)
        # add.append(roi_color)
        # print('len(add)',len(add))
        # boom=np.stack(add)
        # print('stacked',boom.shape)
        # print(roi_color.shape)
        # cv2.imwrite(str(i)+'.png',roi_color)
        # print('------',faces)
        # face_patches = np.stack(faces)
        # print('======',face_patches)

        
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#crop_img = img[y:y+h, x:x+w]
    cv2.imshow('Haarcascades',img)
    # e=time.time()
    # print("TIME :",e-s)
    # i+=1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

print(i)

cap.release()
cv2.destroyAllWindows()


