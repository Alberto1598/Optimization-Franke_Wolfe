import tensorflow as tf
import util
import model


def fast_gradient_sign_method_untargeted(model, x, label, eps = 0.05) :

    loss, gradient, prediction = util.input_network(model, x, label)
    new_image = x + (eps * tf.sign(gradient)) # UNTARGETED 
    new_image = tf.clip_by_value(new_image, 0, 1)
    return new_image


def fast_gradient_sign_method_targeted(model, x, label, eps) :

    loss, gradient, prediction = util.input_network(model, x, label)
    new_image = x - (eps * tf.sign(gradient)) # TARGETED 
    new_image = tf.clip_by_value(new_image, 0, 1)
    return new_image
