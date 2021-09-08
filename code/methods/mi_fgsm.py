import tensorflow as tf
import util


def MI_FGSM_untargeted(model, x, label, label_name, max_iter = 10, eps = 0.05, stepsize = 0.03, mu = 0.9) :
    it = 0
    g = 0
    x_ori = x
    loss = [] 

    while (it < max_iter) :
        l, gradient, prediction = util.input_network(model, x, label)
        if (prediction != label_name) :
          if it==0: 
            print("prediction = ",prediction, " ** label_name = ", label_name)
          break
        loss.append(l)
        new_g = (mu* g) + (gradient / tf.norm(gradient, ord = 1))
        g = new_g
        # clip the new_x inside the bound in order to have || x - x* ||_inf <= eps
        new_x = tf.clip_by_value(x + (stepsize * tf.sign(new_g)), x_ori - eps, x_ori + eps)
        # clip the new_x in order to have value between 0 and 1 for images
        new_x = tf.clip_by_value(new_x, 0, 1) # UNTARGETED 
        x = new_x
        it += 1
    return x, loss, it


def MI_FGSM_targeted(model, x, label, label_name, max_iter = 10, eps = 0.05, stepsize = 0.03, mu = 0.9) :
    it = 0
    g = 0
    x_ori = x
    loss = []
    while (it < max_iter) :
        l, gradient, prediction = util.input_network(model, x, label)
        if (prediction == label_name) :
          break
        loss.append(l)
        new_g = (mu* g) + (gradient / tf.norm(gradient, ord = 1))
        g = new_g
        # clip the new_x inside the bound in order to have || x - x* ||_inf <= eps
        new_x = tf.clip_by_value(x - (stepsize * tf.sign(new_g)), x_ori - eps, x_ori + eps)
        # clip the new_x in order to have value between 0 and 1 for images
        new_x = tf.clip_by_value(new_x, 0, 1) # TARGETED 
        x = new_x
        it += 1
    return x, loss, it
