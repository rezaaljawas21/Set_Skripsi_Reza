# Program still should be evaluated, please evaluate thoroughly, then run in raspberry pi emulator
# check the argparse and check the checking section, then check the under
# open https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html and door_monitor2.py and the tutorial book for confirming
# USAGE
# python door_monitor.py --conf config/config.json

RUSAGE_SELF = 0
# import the necessary packages
from imutils.video import VideoStream
from imutils import paths
from pyimagesearch.utils import Conf
from datetime import datetime
from pathlib import Path
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import signal
import time
import cv2
import sys
import os

#library memory utilization
import psutil

#library resource usage
import resource

original_stdout = sys.stdout
with open("TerminalHasilUjiCoba.txt","w") as f:
  sys.stdout = f
  print("before program is running:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  
  sys.stdout = original_stdout
  # function to handle keyboard interrupt
  def signal_handler(sig, frame):
    print("[INFO] You pressed `ctrl + c`! Closing face recognition" \
			" door monitor application...")
    sys.exit(0)
  sys.stdout = f
  print("before argument parser:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-c", "--conf", required=True,
  help="Path to the input configuration file")
  args = vars(ap.parse_args())
  print(' ')
  print("after argument parser:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  
  # load the configuration file
  conf = Conf(args["conf"])
  print(' ')
  print("after load configuration file:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  
  # load the actual face recognition model, label encoder, and face
  # detector
  videoPaths = []
  path_to_video = "../VideoUji"
  for filename in os.listdir(path_to_video):
    if filename.endswith(".mp4"):
      videoPaths.append(os.path.join(path_to_video, filename))
    else:
      continue
  recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
  le = pickle.loads(open(conf["le_path"], "rb").read())
  detector = cv2.CascadeClassifier(conf["cascade_path"])
  print(' ')
  print("after load model, label encoder, and detector:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  
  # initialize the MOG background subtractor object
  mog = cv2.createBackgroundSubtractorMOG2()
  print(' ')
  print("after initialization of MOG background subtractor object:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  
  # initialize the frame area
  frameArea = None
  
  # initialize previous and current person name to None, then set the
  # consecutive recognition count to zero
  userInputPerson = None
  prevPerson = None
  curPerson = None
  consecCount = 0
  correctCount = 0
  wrongCount = 0
  unknownCount = 0
  correctFlag = None
  person_name = ""

  # initalize the skip frames boolean and skip frane counter
  skipFrames = False
  skipFrameCount = 0
  
  sys.stdout = original_stdout
  # signal trap to handle keyboard interrupt
  signal.signal(signal.SIGINT, signal_handler)
  print("[INFO] Press `ctrl + c` to exit, or 'q' to quit if you have" \
		" the display option on...")
  sys.stdout = f
  for (i, videoPath) in enumerate(videoPaths):
    sys.stdout = original_stdout
    print(videoPath)
    print("[INFO] processing video {}/{}".format(i+1,len(videoPaths)))
    name = videoPath.split(os.path.sep)[-2]
    sys.stdout = f
    cap = cv2.VideoCapture(videoPath)
    #=========Check this line of code==============#
    startTime = datetime.now()
    #==============================================#
    while(cap.isOpened()):
      ret, frame = cap.read()
      # check to see if skip frames is set and the skip frame count is
      # less than the threshold set
      if skipFrames and skipFrameCount < conf["n_skip_frames"]:
        # increment the skip frame counter and continue
        skipFrameCount += 1
        continue
      
      # if the required number of frames have been skipped then reset
      # skip frames boolean and skip frame counter, and reinitialize
      # MOG object
      elif skipFrameCount == conf["n_skip_frames"]:
        skipFrames = False
        skipFrameCount = 0
        mog = cv2.bgsegm.createBackgroundSubtractorMOG()
      
      # resize the frame
      if type(frame) == type(None):
        break
      frame = imutils.resize(frame, width=500)
      print(' ')
      print("after frame resizing:")
      print(psutil.virtual_memory())
      print(resource.getrusage(RUSAGE_SELF))
      
      # if we haven't calculated the frame area yet, calculate it
      if frameArea == None:
        frameArea = (frame.shape[0] * frame.shape[1])
        print(' ')
        print("after calculation of the frame area:")
        print(psutil.virtual_memory())
        print(resource.getrusage(RUSAGE_SELF))
      
      delta = (datetime.now()-startTime).seconds
      if delta <= conf["look_for_a_face"]:
        # convert the input frame from (1) BGR to grayscale (for
        # face # detection) and (2) from BGR to RGB (for face
        # recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(' ')
        print("after converting input frame from 1 BGR to Grayscale and 2 BGR to RGB:")
        print(psutil.virtual_memory())
        print(resource.getrusage(RUSAGE_SELF))
        
        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(' ')
        print("after detecting faces in the grayscale frame:")
        print(psutil.virtual_memory())
        print(resource.getrusage(RUSAGE_SELF))
        
        # OpenCV returns bounding box coordinates in (x, y, w, h)
        # order but we need them in (top, right, bottom, left)
        # order for dlib, so we need to do a bit of reordering
        box = [(y, x + w, y + h, x) for (x, y, w, h) in rects]
        print(' ')
        print("after returning bounding box coordinates:")
        print(psutil.virtual_memory())
        print(resource.getrusage(RUSAGE_SELF))
        
        # check if a face has been detected
        if len(box) > 0:
          # compute the facial embedding for the face
          encodings = face_recognition.face_encodings(rgb, box)
          # perform classification to recognize the face
          preds = recognizer.predict_proba(encodings)[0]
          j = np.argmax(preds)
          curPerson = le.classes_[j]
          # draw the bounding box of the face predicted name on
          # the image
          (top, right, bottom, left) = box[0]
          cv2.rectangle(frame, (left, top), (right,bottom), (0, 255, 0), 2)
          y = top - 15 if top - 15 > 15 else top + 15
          cv2.putText(frame, curPerson, (left, y),
          cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
          
          # if the person recognized is the same as in the
          # previous frame then increment the consecutive count
          if prevPerson == curPerson:
            consecCount += 1
            
          # otherwise, a different name was predicted so reset
          # the counter
          else:
            consecCount = 0
          
          # set current person to previous person for the next
          # iteration
          prevPerson = curPerson
          
          # if a particular person is recognized for a given
          # number of consecutive frames, we have reached a
          # conclusion and alert/greet the person accordingly
          if consecCount == conf["consec_frames"]:
            sys.stdout = original_stdout
            # print the recognized person
            print("[INFO] recognized {}...".format(curPerson))
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            grays = cv2.resize(gray, (960, 540))
            cv2.imshow('frame',grays)
            cv2.waitKey(0)
            sys.stdout = f
            #--------------------------#
            # this line indicated that the person has been recognize
            # this line of code should be replaced with user should
            # confirm if the person showed is correct, if it is indeed 
            # correct then should put in a counter and print the name 
            # if it's false count the false and what the name should be
            person_name = Path(videoPath).stem
            if curPerson == person_name:
              correctFlag = True
              correctCount += 1
            elif curPerson != person_name:
              correctFlag = False
              wrongCount += 1
            print("PythonGuess:{},NameShouldbe:{},{}".format(curPerson,person_name,correctFlag))
            #--------------------------#
            
            # check if the person is an intruder
            if curPerson == "unknown":
              #--------------------------#
              # this line indicated that the person is unknown or fail to recognize
              # this line of code should be replaced with user confirming and count 
              # in other counter indicator for unknown faces
              unknownCount += 1
              print("PythonGuess:{},NameShouldbe:{},{}".format(curPerson,person_name))
              #--------------------------#
            print(' ')
            print("detecting faces:")
            print(psutil.virtual_memory())
            print(resource.getrusage(RUSAGE_SELF))
            
      else:
      #if (delta > conf["look_for_a_face"]) or cv2.waitKey(1)&0xFF == ord('q'):
        if (delta > conf["look_for_a_face"]):
          break
    sys.stdout = original_stdout
    print("[INFO] Next video check")
    skipFrames = False
    skipFrameCount = 0
    sys.stdout = f  
  sys.stdout = f
  print("\nwrong: {}, correct: {}, total: {}".format(wrongCount, correctCount, wrongCount+correctCount))
  cap.release()
  cv2.destroyAllWindows()

### PROGRESS 17:47 18:03:2021,
# i have tidying up the code, still need more reference (please read book that mas dino gave, please seek on the documentation) 
# Configuration and path should be configured further, 
# Bundle up the pickle etc.
# Test first on the google colab and then test on qemu
