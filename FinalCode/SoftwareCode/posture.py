import cv2
import sys
import numpy as np
from time import sleep
import win10toast

#Create the cascade
cascPath = 'C:/Users/AKSHITA GUPTA/Desktop/Postur-corrector/Akshita-posture/haarcascade_frontalface_default.xml';
faceCascade = cv2.CascadeClassifier(cascPath)

#Video Font
font = cv2.FONT_HERSHEY_SIMPLEX

#Windows Notification
toaster = win10toast.ToastNotifier()

class CheckPosture:

    def __init__(self, scale=1, key_points={}):
        self.key_points = key_points
        self.scale = scale
        self.message = ""

    def set_key_points(self, key_points):
        """
        Updates the key_points dictionary used in posture calculations
        :param key_points: {}
            the dictionary to use for key_points
        """
        self.key_points = key_points

    def get_key_points(self):
        """
        Returns the instance's current version of the key_points dictionary
        :return: {}
            the current key_points dictionary
        """
        return self.key_points

    def set_message(self, message):
        """
        Setter to update the message manually if needed
        :param message: string
            The message to override the current message
        """
        self.message = message

    def build_message(self):
        """
        Builds a string with advice to the user on how to correct their posture
        :return: string
            The string containing specific advice
        """
        current_message = ""
        if not self.check_head_drop():
            current_message += "Lift up your head!\n"
        if not self.check_lean_forward():
            current_message += "Lean back!\n"
        if not self.check_slump():
            current_message += "Sit up in your chair, you're slumping!\n"
        self.message = current_message
        return current_message

def findFaces(video_capture):
    """
    Method identifies the cordinates/size of any faces in frame. 
    """
    face = False
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE)
    
    for (x,y,w,h) in faces:
        area = w*h
        #cv2.putText(frame, "Face Area: "+str(area), (8, 465), font,  1, (0,0,255), 1, cv2.LINE_AA)
        face = True
        
    if face:
        return face, frame, area, (x,y,w,h)
    
    elif not face:
        return face, frame, 0, (0,0,0,0)
    
    else:
        return frame
def main():
    # load the configuration data from config.json
    config = load_json(CONFIG_FILE)
    scale = config.get(SCALE)

    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")

    pose_estimator.load(
            engine=edgeiq.Engine.DNN,
            accelerator=edgeiq.Accelerator.CPU)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            posture = CheckPosture(scale)

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = ["Model: {}".format(pose_estimator.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                for ind, pose in enumerate(results.poses):
                    text.append("Person {}".format(ind))
                    text.append('-'*10)
                    text.append("Key Points:")

                    # update the instance key_points to check the posture
                    posture.set_key_points(pose.key_points)

                    correct_posture = posture.correct_posture()
                    if not correct_posture:
                        text.append(posture.build_message())
                        
                        # make a sound to alert the user to improper posture
                        print("\a")

                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
def setAverages():
    """
    Input: None
    Returns: Average Face Area, Average (x, y) coordinate values
    
    Sets the average face area and average (x,y) coordinate values that will be used in 
    further methods
    """
    
    video_capture = cv2.VideoCapture(0)
    areaList = []
    xyList = []
    
    while True:
        face, frame, area, (x,y,w,h) = findFaces(video_capture)
        
        if face:
            areaList.append(area)
            xyList.append([x,y])
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, "Sit with good posture", (8, 465), font,  1, (0,0,255), 1, cv2.LINE_AA)
            
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    averageArea = sum(areaList)/len(areaList)
    xyList = np.asarray(xyList, dtype=np.float32)
    avgXYList = np.mean(xyList, axis=0)
    
    video_capture.release()
    cv2.destroyAllWindows()
    
    return averageArea, avgXYList
    
def createDataset(averageArea, avgXYList):
    """
    Input: Average Face Area, Average (x,y) Coordinate Values
    Returns: Area Change and Distance Change for each frame
    
    Creates a dataset that contains the change in the face's size and how far 
    the face has moved
    
    """    
    areaChangeList = []
    distList = []
    postureList = []
    while True:
        face, frame, area, (x,y,w,h) = findFaces()
        
        if face:
            #areaChange = abs(area - averageArea)
            areaChange =  area - averageArea
            areaChangeList.append(areaChange)
            
            dist = pow((
                        pow((x-avgXYList[0]), 2) +
                        pow((y-avgXYList[1]),2))
                    , 1/2)
            distList.append(dist)
            #1000, 50
            
            if (dist>=50 or areaChange>=1500):
                color = (0,0,255)
                posture = False
            else:
                color = (0,255,0)
                posture = True
            
            postureList.append(posture)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "DistChange: " + str(dist), (8, 465), font,  1, (0,0,255), 1, cv2.LINE_AA)
    
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    
    return areaChangeList, distList, postureList
    
def checkPosture(averageArea, avgXYList):
    """
    Input: Average Face Area, Average (x,y) Coordinate Values
    Returns: Area Change and Distance Change for each frame
    
    Creates a dataset that contains the change in the face's size and how far 
    the face has moved
    
    """    
   
    video_capture = cv2.VideoCapture(0)
    
    face, frame, area, (x,y,w,h) = findFaces(video_capture)
    posture = False
    
    if face:
        areaChange =  area - averageArea
        
        dist = pow((
                    pow((x-avgXYList[0]), 2) +
                    pow((y-avgXYList[1]),2))
                , 1/2)

        if (dist>=50 or areaChange>=1500):
            color = (0,0,255)
            posture = False
        else:
            color = (0,255,0)
            posture = True
                    
  
    video_capture.release()
    return face, posture

def main():
    # load the configuration data from config.json
    config = load_json(CONFIG_FILE)
    scale = config.get(SCALE)

    pose_estimator = edgeiq.PoseEstimation("alwaysai/human-pose")

    pose_estimator.load(
            engine=edgeiq.Engine.DNN,
            accelerator=edgeiq.Accelerator.CPU)

    print("Loaded model:\n{}\n".format(pose_estimator.model_id))
    print("Engine: {}".format(pose_estimator.engine))
    print("Accelerator: {}\n".format(pose_estimator.accelerator))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            posture = CheckPosture(scale)

            # loop detection
            while True:
                frame = video_stream.read()
                results = pose_estimator.estimate(frame)
                # Generate text to display on streamer
                text = ["Model: {}".format(pose_estimator.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))
                for ind, pose in enumerate(results.poses):
                    text.append("Person {}".format(ind))
                    text.append('-'*10)
                    text.append("Key Points:")

                    # update the instance key_points to check the posture
                    posture.set_key_points(pose.key_points)

                    correct_posture = posture.correct_posture()
                    if not correct_posture:
                        text.append(posture.build_message())
                        
                        # make a sound to alert the user to improper posture
                        print("\a")

                streamer.send_data(results.draw_poses(frame), text)

                fps.update()

                if streamer.check_exit():
                    break
averageArea, avgXYList = setAverages()
while True:
    face, posture = checkPosture(averageArea, avgXYList)
    print("Checking")
    if face:
        if not posture:
            toaster.show_toast('Posture Check', "FIX YOUR POSTURE!", duration=5)
        sleep(2) #wait 10 seconds only if there is a face. if not dont wait
    
    
