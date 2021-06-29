# Intruder Detection IoT
 
This project aims to monitor home when owners are away. This project detects any humans in the video feeds and log it.

# Working of the project

There are 2 files client.py which runs on a Raspberry Pi with a camera module (or another computer for testing purposes) and flaskblog.py which runs on a main computer with a reasonable computing power.

client.py captures the video feed from camera and sends it to the main computer using Python's imagezmq library which is used for video streaming with OpenCV.

flaskblog.py creates a webpage at localhost:5000 and waits for connections. When it receives a connection from a Raspberry Pi it prints its address and name in the command prompt. It then runs a Mobilenet SSD model on the video feed. Mobilenet SSD is a Convolutional Neural Network that performs object detection in real time. It requires less computational power compared to other object detection models.

The live video feed with detections is then streamed in the webpage which can be viewed by any device in the local network. If a person is detected, it captures the video and stores it in gif format in /static/Detections folder and also logs it in a text file. 


# How to run

1. Open command prompt and go to the project folder.
2. Run flaskblog.py as follows:

     \>\>\> python flaskblog.py

3. Open another command prompt in same location and run client.py as follows:

     \>\>\> python client.py -s SERVER_ADDRESS

4. SERVER_ADDRESS is the IP address of the computer running flaskblog program.
5. In a browser go to localhost:5000 (on the system running flaskblog) or SERVER_ADDRESS:5000 (on any other device in the local network) to view the live feed along with the detections in logs tab.

# Future work

A loud buzzer can be added to Raspberry Pi and the main computer can send messages to respective Raspberry Pi if a human is detected in its video feed to alert the neighbourhood.

A GSM module can be integrated with Raspberry Pi which can call the local police if human is detected.

