# USAGE
# python create_voice_msgs.py --conf config/config.json

# import the necessary packages
from pyimagesearch.utils import Conf
from gtts import gTTS
import argparse
import pickle
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
	help="Path to the input configuration file")
args = vars(ap.parse_args())

# load the configuration file and label encoder
conf = Conf(args["conf"])
le = pickle.loads(open(conf["le_path"], "rb").read())
print("[INFO] creating mp3 files...")

# loop over all class labels (i.e., names)
for name in le.classes_:
	# display which name we're creating the MP3 for
	print("[INFO] creating {}.mp3...".format(name))

	# if the name is unknown then it's a intruder
	if name == "unknown":
		# initialize the Google Text To Speech object with the
		# message for a intruder
		tts = gTTS(text=conf["intruder_sound"], lang="{}-{}".format(
			conf["lang"], conf["accent"]))

	# otherwise, it's a legitimate person name
	else:
		# initialize the Google Text To Speech object with a welcome
		# message for the person
		tts = gTTS(text="{} {}.".format(conf["welcome_sound"], name),
			lang="{}-{}".format(conf["lang"], conf["accent"]))

	# save the speech generated as a mp3 file
	p = os.path.sep.join([conf["msgs_path"], "{}.mp3".format(name)])
	tts.save(p)
