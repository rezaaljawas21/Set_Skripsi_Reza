# USAGE
# python encode_faces.py --dataset ../../datasets/face_recognition_dataset \
#	--encodings ../output/encodings.pickle

RUSAGE_SELF = 0
# import the necessary packages
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
import imutils

#library memory utilization
import psutil

#library resource usage
import resource

#library to print to file
import sys

original_stdout = sys.stdout
with open("TerminalOutput_Encode.txt","w") as f:
  sys.stdout = f

  print("before argument parser:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))
  # construct the argument parser and parse the arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--dataset", required=True,
	help="path to input directory of faces + images")
  ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
  ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
  args = vars(ap.parse_args())
  print(' ')
  print("after argument parser:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))


  # grab the paths to the input images in our dataset
  sys.stdout = original_stdout
  print("[INFO] quantifying faces...")
  sys.stdout = f
  imagePaths = list(paths.list_images(args["dataset"]))
  print(' ')
  print("after grabing the paths:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))

  # initialize the list of known encodings and known names
  knownEncodings = []
  knownNames = []
  print(' ')
  print("after list initialization:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))

  # loop over the image paths
  for (i, imagePath) in enumerate(imagePaths):
	  # extract the person name from the image path
	  sys.stdout = original_stdout
	  print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	  sys.stdout = f
	  name = imagePath.split(os.path.sep)[-2]
	  print(' ')
	  print("after extracting name {}:".format(i+1,len(imagePaths)))
	  print(psutil.virtual_memory())
	  print(resource.getrusage(RUSAGE_SELF))

	  # load the input image and convert it from RGB (OpenCV ordering)
	  # to dlib ordering (RGB)
	  image = cv2.imread(imagePath)
	  rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	  print(' ')
	  print("after load input image and convert from RGB (Ordering OpenCV) to dlib ordering {}:".format(i+1,len(imagePaths)))
	  print(psutil.virtual_memory())
	  print(resource.getrusage(RUSAGE_SELF))
	
	  # detect the (x, y)-coordinates of the bounding boxes
	  # corresponding to each face in the input image
	  boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])
	  print(' ')
	  print("after detect coordinates of bounding boxes corresponding to each face {}:".format(i+1,len(imagePaths)))
	  print(psutil.virtual_memory())
	  print(resource.getrusage(RUSAGE_SELF))

	  # compute the facial embedding for the face
	  encodings = face_recognition.face_encodings(rgb, boxes)
	  print(' ')
	  print("after computing the facial embedding of each face {}:".format(i+1,len(imagePaths)))
	  print(psutil.virtual_memory())
	  print(resource.getrusage(RUSAGE_SELF))

	  # loop over the encodings
	  for encoding in encodings:
		  # add each encoding + name to our set of known names and
		  # encodings
		  knownEncodings.append(encoding)
		  knownNames.append(name)
		  print(' ')
		  print("after adding each encoding + name to set of known names and encodings{}:".format(i+1,len(imagePaths)))
		  print(psutil.virtual_memory())
		  print(resource.getrusage(RUSAGE_SELF))
	  print(' ')
	  print("after Loop {}:".format(i+1,len(imagePaths)))
	  print(psutil.virtual_memory())
	  print(resource.getrusage(RUSAGE_SELF))
  print(' ')
  print("after processing image:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))

  # dump the facial encodings + names to disk
  sys.stdout = original_stdout
  print("[INFO] serializing encodings...")
  sys.stdout = f
  data = {"encodings": knownEncodings, "names": knownNames}
  f = open(args["encodings"], "wb")
  f.write(pickle.dumps(data))
  f.close()
  print(' ')
  print("after serializing encodings:")
  print(psutil.virtual_memory())
  print(resource.getrusage(RUSAGE_SELF))

sys.stdout = original_stdout