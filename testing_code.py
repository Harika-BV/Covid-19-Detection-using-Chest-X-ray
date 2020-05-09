# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import random
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to out input directory of images")
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
args = vars(ap.parse_args())

# load the pre-trained network
print("[INFO] loading pre-trained network...")
model = load_model(args["model"])

# grab all image paths in the input directory and randomly sample them
imagePaths = list(paths.list_images(args["images"]))
random.shuffle(imagePaths)
imagePaths = imagePaths[:16]
# initialize our list of results
results = []

for p in imagePaths:
	orig = cv2.imread(p)
	image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	pred = model.predict(image)
	pred = pred.argmax(axis=1)[0]
	label = "Covid" if pred == 0 else "Normal"
	color = (0, 0, 255) if pred == 0 else (0, 255, 0)
	orig = cv2.resize(orig, (128, 128))
	cv2.putText(orig, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
		color, 2)
	results.append(orig)
	
montage = build_montages(results, (128, 128), (3, 3))[0]
cv2.imshow("Results", montage)
cv2.waitKey(0)
