import numpy as np
import cv2

faceclassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeclassifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def facedetector(img,size = 0.5):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceclassifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return img
    
    for (x,y,w,h) in  faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img,(x,y),(x+w,y+h),(231,76,85),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        eyes = eyeclassifier.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+w,ey+h),(0,0,231),2)
            
    roi_color = cv2.flip(roi_color,1)
    return roi_color


import urllib.request
url = "http://192.168.0.101:8080/shot.jpg"
while True:
    imgurl = urllib.request.urlopen(url)
    img = np.array(bytearray(imgurl.read()),dtype = np.uint8)
    img = cv2.imdecode(img,-1)
    cv2.imshow("face detection",facedetector(img))
    if cv2.waitKey(1) == 13:
        break
cv2.destroyAllWindows()
