from flask import Flask, render_template, url_for, Response

import imutils
from imutils import build_montages

import time
import datetime
from datetime import datetime
import threading

import cv2
from PIL import Image
import numpy as np

import imagezmq


outputFrame = None
lock = threading.Lock()
threadLock = threading.Lock()

detected = False
old = 0
test = 0

tempDi = dict()
tempDi['title'] = 'Person Detected...'
tempDi['time'] = ''
#tempDi['gif'] = ''

gifImage = []
imageCount = 0
fileName = 0

threadCount = 0

imageHub = imagezmq.ImageHub()

lastActive = {}
lastActiveCheck = datetime.now()


app = Flask(__name__)



posts = []


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/logs")
def logs():
    return render_template('logs.html', posts=posts, title='Logs')



def saveFile():
    global fileName, gifImage, test, threadCount, tempDi, posts

    if(len(gifImage) > 5):
        fileName += 1
        #print("[INFO]  Saving...  Time: {}".format(datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")) )
        directory = 'static/Detections/'+str(fileName)+'.gif'
        #print(directory)
        #print(len(gifImage))
        logFile = open("logs.txt", "a")
        logFile.write((tempDi['time']+" \n"))
        logFile.close()

        #tempDi['gif'] = directory
        posts.append(tempDi.copy())
        tempDi['time'] = ''
    
        gifImage[0].save(directory, save_all=True, append_images=gifImage[1:], optimize=False, loop=0)
        #print("[INFO]  Video saved....")
    gifImage.clear()
    imageCount = 0
    test = 0
    threadCount = 0
    return


def Camera():

    global outputFrame, imageHub, lastActive, lastActiveCheck, posts, old, detected, gifImage, imageCount, test, fileName, threadCount, tempDi


    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')

    
    CONSIDER = "person"
    objCount = 0
    frameDict = {}


    ACTIVE_CHECK_SECONDS = 20

    print("[INFO] detecting person...")

    # start looping over all the frames
    while True:
        
        (rpiName, frame) = imageHub.recv_image()
        imageHub.send_reply(b'OK')

        if rpiName not in lastActive.keys():
            print("[INFO] receiving data from {}...".format(rpiName))

        lastActive[rpiName] = datetime.now()

        actual = frame.copy()
        frame = imutils.resize(frame, width=600)

        #############################################################print(str(test)+' '+str(old))
        if(test == 1):
            cv2.putText(actual, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            pil_image=Image.fromarray(cv2.cvtColor(actual, cv2.COLOR_BGR2RGB))
            if(old > 0 and imageCount < 300):
                #print("if")
                gifImage.append(pil_image)
                imageCount+=1
            else:
                #print("else")
                if(threadCount == 0):
                    threadCount = 1
                    t2 = threading.Thread(target=saveFile)
                    t2.daemon = True
                    t2.start()
                    


        
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)


        net.setInput(blob)
        detections = net.forward()

        # reset the object count for each object in the CONSIDER set
        objCount = 0

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.4:
                # extract the index of the class label from the
                # detections
                idx = int(detections[0, 0, i, 1])

                # check to see if the predicted class is in the set of
                # classes that need to be considered
                if(CLASSES[idx] == CONSIDER):
                    # increment the count of the particular object
                    # detected in the frame
                    objCount += 1

                    if(detected == False):
                        detectedTime = datetime.now()
                        tempDi['time'] = detectedTime.strftime("%A %d %B %Y %I:%M:%S%p")

                        

                        if(test == 0):
                            #print("started")
                            cv2.putText(actual, datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                            pil_image=Image.fromarray(cv2.cvtColor(actual, cv2.COLOR_BGR2RGB))
                            gifImage.append(pil_image)
                            test = 1
                            

                        detected = True

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # draw the bounding box around the detected object on
                    # the frame
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (255, 0, 0), 2)
        old = objCount
        if(objCount == 0):
            detected = False
            #old = objCount['person']

        # draw the sending device name on the frame
        cv2.putText(frame, rpiName, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        

        # draw the object count on the frame
        label = "Person: "+ str(objCount)
        cv2.putText(frame, label, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

        #frame = imutils.resize(frame, width=800)

        with lock:
            outputFrame = frame.copy()



def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")



if __name__ == '__main__':

    t1 = threading.Thread(target=Camera)
    t1.daemon = True
    t1.start()

    

    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
