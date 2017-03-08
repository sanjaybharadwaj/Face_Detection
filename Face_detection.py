import numpy as np
import cv2

def SetClassifiers(face,eye):
  cascade = []
  cascade.append(cv2.CascadeClassifier(face))
  cascade.append(cv2.CascadeClassifier(eye))
  return cascade

def DetectFaceAndEyes(cascade,key):
  cap = cv2.VideoCapture(0)
  while True:
    if key == 1:    
      ret, img = cap.read()
    else:
      img = cv2.imread("face.jpg")
    Process_Image(cascade, img)
    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cv2.destroyAllWindows()

def Process_Image(cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade[0].detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = cascade[1].detectMultiScale(roi_gray)
      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

def Run(face, eye, key):
    cascade = SetClassifiers(face, eye)
    DetectFaceAndEyes(cascade, key)

Run("haarcascade_frontalface_default.xml", "haarcascade_eye.xml", 0)
