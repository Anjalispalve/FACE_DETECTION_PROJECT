#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')

import cv2


# In[10]:


print(cv2.__version__)


# In[7]:


# face_cap = cv2.CascadeClassifier(" C:/Users/hp/AppData/Roaming/Python/Python312/site-packages/cv2/data/haarcascade_frontalface_default.xml")
# video_cap = cv2.VideoCapture(0)
# while True :
#     ret , video_data = video_cap.read()
#     col = cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY)
#     faces = face_cap.detectMultiScale(
#         col,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30,30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
#     for (x,y,w,h) in faces:
#         cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.imshow("video_live",video_data)
#     if cv2.waitKey(10) == ord ("a"):
#         break
# video_cap.release()


# In[4]:


# video_cap = cv2.VideoCapture(0)
# while True :
#     ret , video_data = video_cap.read()
#     cv2.imshow("video_live",video_data)
#     if cv2.waitKey(10) == ord ("a"):
#         break
# video_cap.release()


# In[10]:


# Load Haar cascade
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cap.empty():
    raise IOError("Haar cascade file could not be loaded.")

# Open video capture
video_cap = cv2.VideoCapture(0)
if not video_cap.isOpened():
    raise IOError("Cannot open webcam.")

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Detection', video_data)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()


# In[ ]:




