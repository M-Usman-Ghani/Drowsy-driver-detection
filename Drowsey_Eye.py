import cv2
import numpy as np
from keras.models import load_model



face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

model = load_model('.\models\DrowseyEyeDetection.h5')

leftLabel = ""
rightLabel = ""
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0



while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        if(rpred > 0.5):
            rightLabel='Open' 
            print(rpred)
            print(rightLabel)
        else:
            rightLabel='Closed'
            print(rpred)
            print(rightLabel)
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict(l_eye)
        if(lpred >  0.5):
            leftLabel='Open'
            print(lpred) 
            print(leftLabel)  
        else :
            leftLabel='Closed'
            print(lpred)
            print(leftLabel)
        break


    if(leftLabel == "Closed" and rightLabel == "Closed"):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA) 

    if(score<0):
        score=0   
        cv2.putText(frame,'Score:'+str(score)+' -Awake',(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    elif(score>0 and score<15):
        cv2.putText(frame,'Score:'+str(score)+' -Awake',(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)    
    else:
        cv2.putText(frame,'Score'+str(score)+ ' -Drowsy',(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

    

