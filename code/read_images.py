import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import os
import util

def read_images(model) :
	images_label = pd.read_csv("ImageNet_sum.csv")
	# dataframe containing the name of the image and the associated label
	images_label_df = pd.DataFrame(images_label, columns = ["ImagePath", "class", "name"])
	folder = "./data"
	# create a empty list of images
	images = []
	for filename in os.listdir(folder):
	  # read the image in BGR format
	  img = cv2.imread(os.path.join(folder,filename))
	  # convert to RGB format
	  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	  if img is not None:
	  	img = util.preprocess(img, model) * 0.5 + 0.5
	  	label_index = images_label_df.loc[images_label_df["ImagePath"] == filename]["class"] # returns a pandas series
	  	label_name = images_label_df.loc[images_label_df["ImagePath"] == filename]["name"] # returns a pandas series
	  	img_and_name = (img, label_index.values[-1], label_name.values[-1])
	  	images.append(img_and_name)
	return images


