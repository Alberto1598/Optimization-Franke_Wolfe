import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import cv2
import os
import methods.fgsm as fgsm
from model import Model



def preprocess(image, model):

  #  Resnet152 needs a different preprocess 

  image = tf.cast(image, tf.float32)
  if model.name == 'mobilenet_v2':
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  elif model== 'resnetV2': 
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
  else:
    image = tf.image.resize(image, (299, 299))
    image = tf.keras.applications.inception_v3.preprocess_input(image)

  
  image = image[None, ...]
  return image



# Using gradient tape to watch the input image through the model 
# and being able to have the gradient of the loss wrt input image 

def compute_gradient(input_image, input_label, loss_object, model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model.pretrained_model(input_image)
        loss = loss_object(input_label, prediction)
   # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    return loss, gradient



# Input the image to the model and return the gradient values 

def input_network(model, image, index) :
    
    # 208 for labrador 

    image_probs = model.pretrained_model.predict(image)
    prediction = model.get_imagenet_label(image_probs)
    label = tf.one_hot(index, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))
    loss, gradient = compute_gradient(image, label, tf.keras.losses.CategoricalCrossentropy(), model)
    return loss, gradient, prediction[1]




# divided FGSM from others, since different #parameters in input and output 
# other parameters (such as momentum and beta) are defined inside the corresponding functions 

def untargeted(images, model, function, epsilon, stepsize=0.03, maxit = 20): 
  '''
  AS INPUT: 
  - images: list of images 
  - pretrained_model: like IncV3 or ResNetV2 etc 
  - function: the function we want to run, like fgsm etc 
  - epsilon: the max distortion 
  - maxit: (if not fgsm) the maximum #iterations 
  - stepsize: (if not fgsm) 

  AS OUTPUT: 
  - successes: number of misclassifications
  - distortion_l: list of distortions of the images, ||.||_inf 
  - confidence_l: list of confidence of classifications 
  - it_l: (if not fgsm) list of needed iterations 
  '''

  successes = 0
  loss = []
  distortion_l = []
  confidence_l = []
  it_l = []


  if function == fgsm.fast_gradient_sign_method_untargeted: 
    for i in range(len(images)) :
      # predict the label for the image
      old_image_probs = model.pretrained_model.predict(images[i][0])
      # get the label and the respective class
      _, old_image_class, old_class_confidence = model.get_imagenet_label(old_image_probs)
      # get the loss and the gradient for the image
      loss, gradient, prediction = input_network(model, images[i][0], images[i][1])
      # create the new noisy image
      new_image = function(model, images[i][0], images[i][1], epsilon) 
      # predict the label for the noisy image
      new_image_probs = model.pretrained_model.predict(new_image)
      # get the label and the confidence for the noisy image
      _, new_image_class, new_class_confidence = model.get_imagenet_label(new_image_probs)
      if ( old_image_class != new_image_class) :
        successes += 1
        # compute distortion
        distortion = tf.norm(images[i][0] - new_image, np.inf)
        distortion_l.append(distortion)
        # add class confidence
        confidence_l.append(new_class_confidence)
      print("sample {} : [true label : {} with confidence {:.2f} %] , [predicted label : {} with confidence {:.2f}%] ".format(i, old_image_class, old_class_confidence * 100, new_image_class, new_class_confidence*100))
    print("---SUCCESS RATE : {:.2f} ---".format((successes * 100)/len(images)))
    if distortion_l :
      print("---DISTORSION : {:.2f} ---".format(sum(distortion_l)/ len(distortion_l)))

    return successes, distortion_l, confidence_l

  else: 

      for i in range(len(images)) :
        # predict the label for the image
        old_image_probs = model.pretrained_model.predict(images[i][0])
        # get the label and the respective class
        _, old_image_class, old_class_confidence = model.get_imagenet_label(old_image_probs) 
        new_image, loss, it = function(model, images[i][0], images[i][1], old_image_class, maxit, epsilon, stepsize)

        # append number of iterations 
        it_l.append(it)
        # predict the label for the noisy image
        new_image_probs = model.pretrained_model.predict(new_image)
        # get the label and the confidence for the noisy image
        _, new_image_class, new_class_confidence = model.get_imagenet_label(new_image_probs)
        if ( old_image_class != new_image_class) :
          successes += 1
          # compute distortion
          distortion = tf.norm(images[i][0] - new_image, np.inf)
          distortion_l.append(distortion)
          confidence_l.append(new_class_confidence)
        print("sample {} : [true label : {} with confidence {:.2f} %] , [predicted label : {} with confidence {:.2f}%] ".format(i, old_image_class, old_class_confidence * 100, new_image_class, new_class_confidence*100))
      print("---SUCCESS RATE : {:.2f} ---".format((successes * 100)/len(images)))
      if distortion_l :
        print("---DISTORSION : {:.4f} ---".format(sum(distortion_l)/ len(distortion_l)))
      if it_l :
        print("---AVG NUMBER OF ITERATIONS : {:.2f} ---".format(sum(it_l)/ len(it_l)))

      return successes, distortion_l, confidence_l, it_l 


def targeted(images, model, function, label_index, label_class, epsilon, stepsize = 0.03, maxit = 20): 
  '''
  AS INPUT: 
  - images: list of images 
  - pretrained_model: like IncV3 or ResNetV2 etc 
  - function: the function we want to run, like fgsm etc 
  - label_index: the target we want, as integer index 
  - label_class: the target we want, as string name 
  - epsilon: the max distortion 
  - maxit: (if not fgsm) the maximum #iterations 
  - stepsize: (if not fgsm)

  AS OUTPUT: 
  - successes: number of misclassifications
  - distortion_l: list of distortions of the images, ||.||_inf 
  - confidence_l: list of confidence of classifications 
  - it_l: (if not fgsm) list of needed iterations 
  '''

  successes = 0
  loss = []
  distortion_l = []
  confidence_l = []
  it_l = []


  if function == fgsm.fast_gradient_sign_method_targeted: 
    for i in range(len(images)) :
      # predict the label for the image
      old_image_probs = model.pretrained_model.predict(images[i][0])
      # get the label and the respective class
      _, old_image_class, old_class_confidence = model.get_imagenet_label(old_image_probs)
      # get the loss and the gradient for the image
      loss, gradient, prediction = input_network(model, images[i][0], images[i][1])
      # create the new noisy image
      new_image = function(model, images[i][0], label_index, epsilon) 
      # predict the label for the noisy image
      new_image_probs = model.pretrained_model.predict(new_image)
      # get the label and the confidence for the noisy image
      _, new_image_class, new_class_confidence = model.get_imagenet_label(new_image_probs)
      if (  new_image_class == label_class) : 
        successes += 1
        # compute distortion
        distortion = tf.norm(images[i][0] - new_image, np.inf)
        distortion_l.append(distortion)
        confidence_l.append(new_class_confidence)
      print("sample {} : [true label : {} with confidence {:.2f} %] , [predicted label : {} with confidence {:.2f}%] ".format(i, old_image_class, old_class_confidence * 100, new_image_class, new_class_confidence*100))
    print("---SUCCESS RATE : {:.2f} ---".format((successes * 100)/len(images)))
    if distortion_l: 
      print("---DISTORSION : {:.2f} ---".format(sum(distortion_l)/ len(distortion_l)))
    

    return successes, distortion_l, confidence_l

  else: 

      for i in range(len(images)) :
        # predict the label for the image
        old_image_probs = model.pretrained_model.predict(images[i][0])
        # get the label and the respective class
        _, old_image_class, old_class_confidence = model.get_imagenet_label(old_image_probs)
        # create the new noisy image
        new_image, loss, it = function(model, images[i][0], label_index, label_class, maxit, epsilon, stepsize)
        
        # append number of iterations 
        it_l.append(it)
        # predict the label for the noisy image
        new_image_probs = model.pretrained_model.predict(new_image)
        # get the label and the confidence for the noisy image
        _, new_image_class, new_class_confidence = model.get_imagenet_label(new_image_probs)
        if ( label_class == new_image_class) :
          successes += 1
          # compute distortion
          distortion = tf.norm(images[i][0] - new_image, np.inf)
          distortion_l.append(distortion)
          confidence_l.append(new_class_confidence)
        print("sample {} : [true label : {} with confidence {:.2f} %] , [predicted label : {} with confidence {:.2f}%] ".format(i, old_image_class, old_class_confidence * 100, new_image_class, new_class_confidence*100))
      print("---SUCCESS RATE : {:.2f} ---".format((successes * 100)/len(images)))
      if distortion_l :
        print("---DISTORSION : {:.4f} ---".format(sum(distortion_l)/ len(distortion_l)))
      if it_l :
        print("---AVG NUMBER OF ITERATIONS : {:.2f} ---".format(sum(it_l)/ len(it_l)))

      return successes, distortion_l, confidence_l, it_l 


def grid_search_untargeted(images, model, function, epsilon, stepsize = None, max_it = 20) :
  eps_l = np.linspace(0.01, 0.3, 10)
  it_l = np.arange(2, 22, 2)
  successes_l_untargeted_eps = []
  distortion_l_untargeted_eps = []
  successes_l_untargeted_it = []
  distortion_l_untargeted_it = []
  if function == fgsm.fast_gradient_sign_method_untargeted :
    for eps in eps_l :
      successes, distortion_l, confidence_l = untargeted(images, model, function, eps)
      successes_rate = successes * 100 /len(images)
      if distortion_l :
        distortion_rate = sum(distortion_l) / len(distortion_l)
      else :
        distortion_rate = 0
      successes_l_untargeted_eps.append(successes_rate)
      distortion_l_untargeted_eps.append(distortion_rate)
    return successes_l_untargeted_eps, distortion_l_untargeted_eps
  else :
     for eps in eps_l :
       successes, distortion_l, confidence_l, it_list = untargeted(images, model, function,  eps, stepsize, max_it)
       successes_rate = successes * 100 /len(images)
       if distortion_l :
         distortion_rate = sum(distortion_l) / len(distortion_l)
       else :
         distortion_rate = 0
       successes_l_untargeted_eps.append(successes_rate)
       distortion_l_untargeted_eps.append(distortion_rate)
     for it in it_l :
       successes, distortion_l, confidence_l, it_list = untargeted(images, model, function,  epsilon, stepsize, it)
       successes_rate = successes * 100 /len(images)
       if distortion_l :
        distortion_rate = sum(distortion_l) / len(distortion_l)
       else :
        distortion_rate = 0
       successes_l_untargeted_it.append(successes_rate)
       distortion_l_untargeted_it.append(distortion_rate)
     return successes_l_untargeted_eps, distortion_l_untargeted_eps, successes_l_untargeted_it, distortion_l_untargeted_it


def grid_search_targeted(images, model, function, label_index, label_class, epsilon, stepsize = None, max_it = 20) :
  eps_l = np.linspace(0.01, 0.3, 10)
  it_l = np.arange(2, 22, 2)
  successes_l_targeted_eps = []
  distortion_l_targeted_eps = []
  successes_l_targeted_it = []
  distortion_l_targeted_it = []
  if function == fgsm.fast_gradient_sign_method_targeted :
    for eps in eps_l :
      successes, distortion_l, confidence_l = targeted(images, model, function, label_index, label_class,  eps)
      successes_rate = successes * 100 /len(images)
      if distortion_l :
        distortion_rate = sum(distortion_l) / len(distortion_l)
      else :
        distortion_rate = 0
      successes_l_targeted_eps.append(successes_rate)
      distortion_l_targeted_eps.append(distortion_rate)
    return successes_l_targeted_eps, distortion_l_targeted_eps
  else :
     for eps in eps_l :
       successes, distortion_l, confidence_l, it_list = targeted(images, model, function, label_index, label_class, eps, stepsize, max_it)
       successes_rate = successes * 100 /len(images)
       if distortion_l :
         distortion_rate = sum(distortion_l) / len(distortion_l)
       else :
         distortion_rate = 0
       successes_l_targeted_eps.append(successes_rate)
       distortion_l_targeted_eps.append(distortion_rate)
     for it in it_l :
       successes, distortion_l, confidence_l, it_list = targeted(images, model, function, label_index, label_class, epsilon, stepsize, it)
       successes_rate = successes * 100 /len(images)
       if distortion_l :
         distortion_rate = sum(distortion_l) / len(distortion_l)
       else :
         distortion_rate = 0
       successes_l_targeted_it.append(successes_rate)
       distortion_l_targeted_it.append(distortion_rate)
     return successes_l_targeted_eps, distortion_l_targeted_eps, successes_l_targeted_it, distortion_l_targeted_it


# function involved in plotting graph for grid search

def plot_graphs(targeted, successes_FGSM, successes_eps_PGD, successes_eps_MI, successes_eps_FW, distortion_FGSM,
  distortion_eps_PGD, distortion_eps_MI, distortion_eps_FW, successes_it_PGD, successes_it_MI, successes_it_FW, distortion_it_PGD, distortion_it_MI, distortion_it_FW) :
  eps_l = np.linspace(0.01, 0.3, 10)
  it_l = np.arange(2, 22, 2)

  print(successes_it_PGD)
  print("------")
  print(successes_it_MI)
  print("------")
  print(successes_it_FW)
  subplots, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(20, 10))
  subplots.subplots_adjust(wspace = 0.3, hspace = 0.3)
  ax1.plot(eps_l, successes_FGSM, color = "blue", marker = "o" , label = "FGSM")
  ax1.plot(eps_l, successes_eps_PGD, color = "red", marker = "s", label = "PGD")
  ax1.plot(eps_l, successes_eps_MI, color = "green", marker = "D", label = "MI-FGSM")
  ax1.plot(eps_l, successes_eps_FW, color = "violet", marker = "*", label = "FW-WHITE")
  ax1.set_xlabel("epsilon")
  ax1.set_ylabel("success rate")
  ax1.legend()

  ax2.plot(eps_l, distortion_FGSM, color = "blue", marker = "o" , label = "FGSM")
  ax2.plot(eps_l, distortion_eps_PGD, color = "red", marker = "s", label = "PGD")
  ax2.plot(eps_l, distortion_eps_MI, color = "green", marker = "D", label = "MI-FGSM")
  ax2.plot(eps_l, distortion_eps_FW, color = "violet", marker = "*", label = "FW-WHITE")
  ax2.set_xlabel("epsilon")
  ax2.set_ylabel("distortion rate")
  ax2.legend()

  ax1.set_title("epsilon vs success rate")
  ax2.set_title("epsilon vs distortion rate")

  ax3.plot(it_l, successes_it_PGD, color = "red", marker = "s", label = "PGD")
  ax3.plot(it_l, successes_it_MI, color = "green", marker = "D", label = "MI-FGSM")
  ax3.plot(it_l, successes_it_FW, color = "violet", marker = "*", label = "FW-WHITE")
  ax3.set_xlabel(" max iterations")
  ax3.set_ylabel("success rate")
  ax3.legend()

  ax4.plot(it_l, distortion_it_PGD, color = "red", marker = "s", label = "PGD")
  ax4.plot(it_l, distortion_it_MI, color = "green", marker = "D", label = "MI-FGSM")
  ax4.plot(it_l, distortion_it_FW, color = "violet", marker = "*", label = "FW-WHITE")
  ax4.set_xlabel(" max iterations")
  ax4.set_ylabel("distortion rate")
  ax4.legend()

  ax3.set_title("number of iterations vs success rate")
  ax4.set_title("number of iterations vs distortion rate")

  if (targeted) :
    plt.suptitle("TARGETED ATTACKS")
    plt.savefig("targeted_grid.pdf")
  else :
    plt.suptitle("UNTARGETED ATTACKS")
    plt.savefig("untargeted_grid.pdf")