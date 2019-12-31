import cv2

facecasc = cv2.CascadeClassifier("C:\\Users\\Pradeep\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")
eyecasc= cv2.CascadeClassifier("C:\\Users\\Pradeep\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data\\haarcascade_eye.xml")
videocapture= cv2.VideoCapture(0)

while True:
    ret,frame = videocapture.read()
    grayimg= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = facecasc.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,flags= cv2.CASCADE_SCALE_IMAGE,minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(250,0,0),2)
        roi_gray= grayimg[y:y+h, x:x+w]
        roi_color= frame[y:y+h, x:x+w]
        eyes=eyecasc.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(51,255,51),2)

    cv2.imshow("Video",frame)

    if cv2.waitKey(1) & 0xFF == ord('c'):
        break

videocapture.release()
cv2.destroyAllWindows()