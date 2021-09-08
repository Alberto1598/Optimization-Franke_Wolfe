import tensorflow as tf
import matplotlib.pyplot as plt
import methods.fgsm as fgsm 
import methods.pgd as pgd
import methods.mi_fgsm as mi_fgsm
import methods.fw_white as fw_white
import numpy as np
import util
from model import Model


# function for plotting images
def plot_image(image, prediction, title_image, distortion) :
	plt.figure()
	plt.imshow(preprocessed_image[0])  # To change [-1, 1] to [0,1]
	_, image_class, class_confidence = model.get_imagenet_label(prediction)
	plt.title('{} : {:.2f}% Confidence \n distortion = {:.4f}'.format(image_class, class_confidence*100, distortion))
	plt.savefig(title_image)


# define what model to use
model = Model(tf.keras.applications.InceptionV3(include_top=True,
													 weights='imagenet'), tf.keras.applications.inception_v3.decode_predictions, "inception_v3" )


# class index of the correct label
index = 985
# define the path of the raw image
image_path = tf.keras.utils.get_file('sunflower.jpg', "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg")
image_raw = tf.io.read_file(image_path)
image = tf.image.decode_image(image_raw)
preprocessed_image = util.preprocess(image, model) * 0.5 + 0.5


prediction = model.pretrained_model.predict(preprocessed_image)



plot_image(preprocessed_image, prediction, "img/original_image.pdf", 0)


# FGSM

new_image = fgsm.fast_gradient_sign_method_untargeted(model, preprocessed_image, index)
distortion = tf.norm(preprocessed_image - new_image, np.inf)
prediction = model.pretrained_model.predict(new_image)
print ( "----- FGSM -----")
print(" distortion : {}".format(distortion))
plot_image(model, prediction, "img/fgsm.pdf" , distortion)


# PGD

new_image, loss, it = pgd.PGD_untargeted(model, preprocessed_image, index, "daisy")
prediction = model.pretrained_model.predict(new_image)
distortion = tf.norm(preprocessed_image - new_image, np.inf)
print ( "----- PGD -----")
print (" iteration : {}".format(it))
print(" distortion : {}".format(distortion))
plot_image(model, prediction, "img/pgd.pdf" , distortion)

# MI-FGSM

new_image, loss, it = mi_fgsm.MI_FGSM_untargeted(model, preprocessed_image, index, "daisy")
prediction = model.pretrained_model.predict(new_image)
distortion = tf.norm(preprocessed_image - new_image, np.inf)
print ( "----- MI-FGSM -----")
print (" iteration : {}".format(it))
print(" distortion : {}".format(distortion))
plot_image(model, prediction, "img/mi_fgsm.pdf" , distortion)


# FW-WHITE
new_image, loss, it = fw_white.FW_white_untargeted(model, preprocessed_image, index, "daisy")
prediction = model.pretrained_model.predict(new_image)
distortion = tf.norm(preprocessed_image - new_image, np.inf)
print ( "----- FW-WHITE -----")
print (" iteration : {}".format(it))
print(" distortion : {}".format(distortion))
plot_image(model, prediction, "img/fw_white.pdf" , distortion)