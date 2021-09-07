import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import time
import read_images
import methods.fgsm as fgsm
import methods.pgd as pgd
import methods.mi_fgsm as mi_fgsm
import methods.fw_white as fw_white
import util
from model import Model

if (sys.argv[1] == "inception_v3") :
	model = Model(tf.keras.applications.InceptionV3(include_top=True,
													 weights='imagenet'), tf.keras.applications.inception_v3.decode_predictions, "inception_v3" )


elif (sys.argv[1] == "resnet_v2") :
	model = Model(tf.keras.applications.InceptionResNetV2(include_top=True,
													 weights='imagenet'), tf.keras.applications.inception_resnet_v2.decode_predictions, "resnet_v2")



elif (sys.argv[1] == "mobilenet_v2") :
	model = Model(tf.keras.applications.MobileNetV2(include_top=True,
													 weights='imagenet'), tf.keras.applications.mobilenet_v2.decode_predictions, "mobilenet_v2")

else :

   raise ValueError("Wrong model name")


images = read_images.read_images(model)
eps = 0.05
max_it = 25
stepsize = 0.03

if (sys.argv[3] == "untargeted") :

	if (sys.argv[2] == "fgsm") :
		print( "-----untargeted attacks using FGSM -------")
		successes, distortion_l, confidence_l = util.untargeted(images, model, fgsm.fast_gradient_sign_method_untargeted, eps)
	
	elif (sys.argv[2] == "pgd") :
		print( "-----untargeted attacks using PGD -------")
		successes, distortion_l, confidence_l, it_l = util.untargeted(images, model, pgd.PGD_untargeted,eps, stepsize, max_it)

	elif (sys.argv[2] == "mi-fgsm") :
		print( "-----untargeted attacks using MI-FGSM -------")
		successes, distortion_l, confidence_l, it_l = util.untargeted(images, model, mi_fgsm.MI_FGSM_untargeted, eps, stepsize, max_it)

	elif (sys.argv[2] == "fw-white") :
		stepsize = 0.1
		print( "-----untargeted attacks using FW-WHITE -------")
		successes, distortion_l, confidence_l, it_l = util.untargeted(images, model, fw_white.FW_white_untargeted, eps, stepsize, max_it)

	elif (sys.argv[2] == "grid_search") :
		successes_FGSM, distortion_FGSM = util.grid_search_untargeted(images, model, fgsm.fast_gradient_sign_method_untargeted, 0.05)
		successes_eps_PGD, distortion_eps_PGD, successes_it_PGD, distortion_it_PGD = util.grid_search_untargeted(images, model, pgd.PGD_untargeted, 0.05, 0.03)
		successes_eps_MI, distortion_eps_MI, successes_it_MI, distortion_it_MI = util.grid_search_untargeted(images, model, mi_fgsm.MI_FGSM_untargeted, 0.05, 0.03)
		successes_eps_FW, distortion_eps_FW, successes_it_FW, distortion_it_FW, = util.grid_search_untargeted(images, model, fw_white.FW_white_untargeted, 0.05, 0.1)
		util.plot_graphs(False, successes_FGSM, successes_eps_PGD, successes_eps_MI, successes_eps_FW, distortion_FGSM, distortion_eps_PGD, distortion_eps_MI, distortion_eps_FW,
			successes_it_PGD, successes_it_MI, successes_it_FW, distortion_it_PGD, distortion_it_MI, distortion_it_FW)
	else :

		raise ValueError("Wrong command syntax. Specify the type of optimization method you want to use")

elif (sys.argv[3] == "targeted") :

	label_index = 150
	label_class = "sea_lion"

	if (sys.argv[2] == "fgsm") :
		print( "-----targeted attacks using FGSM -------")
		successes, distortion_l, confidence_l = util.targeted(images, model, fgsm.fast_gradient_sign_method_targeted, label_index, label_class, eps)
	
	elif (sys.argv[2] == "pgd") :
		print( "-----targeted attacks using PGD -------")
		successes, distortion_l, confidence_l, it_l = util.targeted(images, model, pgd.PGD_targeted, label_index, label_class, eps, stepsize, max_it)

	elif (sys.argv[2] == "mi-fgsm") :
		print( "-----targeted attacks using MI-FGSM -------")
		successes, distortion_l, confidence_l, it_l = util.targeted(images, model, mi_fgsm.MI_FGSM_targeted, label_index, label_class, eps, stepsize, max_it)

	elif(sys.argv[2] == "fw-white") :
		stepsize = 0.1
		print( "-----targeted attacks using FW-WHITE -------")
		successes, distortion_l, confidence_l, it_l = util.targeted(images, model, fw_white.FW_white_targeted, label_index, label_class, eps, stepsize, max_it)

	elif(sys.argv[2] == "grid_search") :
		label_name = 150
		label_class = "sea_lion"
		successes_FGSM, distortion_FGSM = util.grid_search_targeted(images, model, fgsm.fast_gradient_sign_method_targeted, label_name, label_class, 0.05)
		successes_eps_PGD, distortion_eps_PGD, successes_it_PGD, distortion_it_PGD = util.grid_search_targeted(images, model, pgd.PGD_targeted, label_name, label_class, 0.05, 0.03)
		successes_eps_MI, distortion_eps_MI, successes_it_MI, distortion_it_MI = util.grid_search_targeted(images, model, mi_fgsm.MI_FGSM_targeted, label_name, label_class, 0.05, 0.03)
		successes_eps_FW, distortion_eps_FW, successes_it_FW, distortion_it_FW, = util.grid_search_targeted(images, model, fw_white.FW_white_targeted, label_name, label_class, 0.05, 0.1)
		util.plot_graphs(True, successes_FGSM, successes_eps_PGD, successes_eps_MI, successes_eps_FW, distortion_FGSM, distortion_eps_PGD, distortion_eps_MI, distortion_eps_FW,
			successes_it_PGD, successes_it_MI, successes_it_FW, distortion_it_PGD, distortion_it_MI, distortion_it_FW)
	else :

		raise ValueError("Wrong command syntax. Specify the type of optimization method you want to use")

else :

	raise ValueError("wrong command syntax. Specify if you want to perform targeted or untargeted attacks")
