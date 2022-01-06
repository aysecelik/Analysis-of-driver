import cv2
import numpy as np
import vlc
import time

net = cv2.dnn.readNet('yolov3_custom_last.weights', 'yolov3_custom.cfg')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
music = vlc.MediaPlayer("C:/Users/HP/Desktop/sürücü analiz/music.mp3")

while True:
    _, img = cap.read()

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,0), 1)
            if(class_ids[i]==0):              
                gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale (gray, 1.3, 5) 
                for (x, y, w, h) in faces:
                    cv2.rectangle (img, (x-50, y-50), (x + w+50, y + 50+h), (0, 0,255), 3)
                    cv2.putText (img, "warning", (x+50, y-50), font, 2, (255, 255, 255), 3)          
                music.play()
                time.sleep(2)
                music.stop()
            
            
            
            
            
            
                
            


    cv2.imshow('Image', img)

    if cv2.waitKey (1) & 0xFF == ord ('q'):
        break
cap.release()
cv2.destroyAllWindows()
