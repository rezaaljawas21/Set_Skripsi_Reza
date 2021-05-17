# USAGE
# python train_model.py --encodings ../output/encodings.pickle \
#	--recognizer ../output/recognizer.pickle --le ../output/le.pickle

RUSAGE_SELF = 0
# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import argparse
import pickle
#library memory utilization
import psutil

#library resource usage
import resource

#library to print to file
import sys

original_stdout = sys.stdout
with open('TerminalOutput_Training.txt','w') as f:
	sys.stdout = f

	print("before argument parser:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--encodings", required=True,
		help="path to serialized db of facial encodings")
	ap.add_argument("-r", "--recognizer", required=True,
		help="path to output model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True,
		help="path to output label encoder")
	args = vars(ap.parse_args())
	print(' ')
	print("after argument parser:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))

	# load the face encodings
	sys.stdout = original_stdout
	print("[INFO] loading face encodings...")
	sys.stdout = f
	data = pickle.loads(open(args["encodings"], "rb").read())
	print(' ')
	print("after loading face encodings:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))

	# encode the labels
	sys.stdout = original_stdout
	print("[INFO] encoding labels...")
	sys.stdout = f
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])
	print(' ')
	print("after encode the label:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))

	# train the model used to accept the 128-d encodings of the face and
	# then produce the actual face recognition
	sys.stdout = original_stdout
	print("[INFO] training model...")
	sys.stdout = f
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["encodings"], labels)
	print(' ')
	print("after training model:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))

	# write the actual face recognition model to disk
	f = open(args["recognizer"], "wb")
	f.write(pickle.dumps(recognizer))
	f.close()
	print(' ')
	print("after writing actual face recognition model to disk:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))

	# write the label encoder to disk
	f = open(args["le"], "wb")
	f.write(pickle.dumps(le))
	f.close()
	print(' ')
	print("after writing label encoder to disk:")
	print(psutil.virtual_memory())
	print(resource.getrusage(RUSAGE_SELF))
	sys.stdout = original_stdout