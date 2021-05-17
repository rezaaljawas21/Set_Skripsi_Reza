# USAGE
# python door_monitor.py --conf config/config.json

# import the necessary packages
from pyimagesearch.notifications import TwilioNotifier
from imutils.video import VideoStream
from pyimagesearch.utils import Conf
from datetime import datetime
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

# function to handle keyboard interrupt
def signal_handler(sig, frame):
	print("[INFO] You pressed `ctrl + c`! Closing face recognition" \
		" door monitor application...")
	sys.exit(0)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and initialize the Twilio notifier
conf = Conf(args["conf"])
tn = TwilioNotifier(conf)

# load the actual face recognition model, label encoder, and face
# detector
recognizer = pickle.loads(open(conf["recognizer_path"], "rb").read())
le = pickle.loads(open(conf["le_path"], "rb").read())
detector = cv2.CascadeClassifier(conf["cascade_path"])

# initialize the MOG background subtractor object
mog = cv2.bgsegm.createBackgroundSubtractorMOG()

# initialize the frame area and boolean used to determine if the door
# is open or closed
frameArea = None
doorOpen = False

# initialize previous and current person name to None, then set the
# consecutive recognition count to zero
prevPerson = None
curPerson = None
consecCount = 0

# initialize the skip frames boolean and skip frame counter
skipFrames = False
skipFrameCount = 0

# signal trap to handle keyboard interrupt
signal.signal(signal.SIGINT, signal_handler)
print("[INFO] Press `ctrl + c` to exit, or 'q' to quit if you have" \
	" the display option on...")

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] warming up camera...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames of the stream
while True:
	# grab the next frame from the stream
	frame = vs.read()

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
	frame = imutils.resize(frame, width=500)

	# if we haven't calculated the frame area yet, calculate it
	if frameArea == None:
		frameArea = (frame.shape[0] * frame.shape[1])

	# if the door is closed, monitor the door using background
	# subtraction
	if not doorOpen:
		# convert the frame to grayscale and smoothen it using a
		# gaussian kernel
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (13, 13), 0)

		# calculate the mask using MOG background subtractor
		mask = mog.apply(gray)

		# find countours in the mask
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# check to see if at least one contour is found
		if len(cnts) >= 1:
			# sort the contours in descending order based on their
			# area and grab the largest one
			c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

			# if the *percentage* of contour area w.r.t. frame to
			# greater than the threshold set then set the door as
			# open and record the start time of this event
			if (cv2.contourArea(c) / frameArea) >= conf["threshold"]:
				print("[INFO] door is open...")
				doorOpen = True
				startTime = datetime.now()

	# if the door is open then:
	# 1) run face recognition for a pre-determined period of time
	# 2) if no face is detected in step 1 then it's a intruder
	elif doorOpen:
		# compute the number of seconds difference between the current
		# timestamp and when the motion threshold was triggered
		delta = (datetime.now() - startTime).seconds

		# run face recognition for pre-determined period of time
		if delta <= conf["look_for_a_face"]:
			# convert the input frame from (1) BGR to grayscale (for
			# face # detection) and (2) from BGR to RGB (for face
			# recognition)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# detect faces in the grayscale frame
			rects = detector.detectMultiScale(gray, scaleFactor=1.1,
				minNeighbors=5, minSize=(30, 30))

			# OpenCV returns bounding box coordinates in (x, y, w, h)
			# order but we need them in (top, right, bottom, left)
			# order for dlib, so we need to do a bit of reordering
			box = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

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
				cv2.rectangle(frame, (left, top), (right,
					bottom), (0, 255, 0), 2)
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
					# play the MPL# file according to the recognized
					# person
					print("[INFO] recognized {}...".format(curPerson))
					os.system("mpg321 --stereo {}/{}.mp3".format(
						conf["msgs_path"], curPerson))

					# check if the person is an intruder
					if curPerson == "unknown":
						# send the frame via Twilio to the home owner
						tn.send(frame)

					# mark the door as closed and now we start skipping
					# next few frames
					print("[INFO] door is closed...")
					doorOpen = False
					skipFrames = True

		# otherwise, no face was detected and the door was closed
		else:
			# indicate the door is not open and then start skipping
			# frames
			print("[INFO] no face detected...")
			doorOpen = False
			skipFrames = True

	# check to see if we should display the frame to our screen
	if conf["display"]:
		# show the frame and record any keypresses
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key is pressed, break from the loop
		if key == ord("q"):
			break

# do a bit of cleanup
vs.stop()