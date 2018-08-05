
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


import cv2


# # Face Detection

# In[5]:


faceclassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread("deepak.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = faceclassifier.detectMultiScale(gray,1.3,5)

if faces is ():
    print("No Face")
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,89,102),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()


# ## Eye Detection with Face

# In[7]:


faceclassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
image = cv2.imread("deepak.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = faceclassifier.detectMultiScale(gray,1.3,5)
eyeclassifier = cv2.CascadeClassifier('haarcascade_eye.xml')
if faces is ():
    print("No Face")

    
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,89,102),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = image[y:y+h,x:x+w]
    eyes = eyeclassifier.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex + ew,ey+ eh),(255,255,0),2)
        cv2.imshow("img",image)
        cv2.waitKey()
    
cv2.destroyAllWindows()


# In[8]:


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

cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    cv2.imshow("Our Face Extractor",facedetector(frame))
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
cv2.destroyAllWindows()
    











