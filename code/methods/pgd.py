import tensorflow as tf
import util
from model import Model


def PGD_untargeted(model, x, label,label_name, max_it=10, epsilon=0.5, s=0.03, alpha=1): 
  x_ori = x 
  it = 0
  loss = []

  for it in range(max_it): 
    l, gradient, prediction = util.input_network(model, x, label)
    if (prediction != label_name) :
          break
    loss.append(l)
    tmp = x + s*gradient # UNTARGETED 

    # "projection"
    # clip because we consider inf norm 
    #tmp = tf.clip_by_value(tmp, -epsilon,epsilon)

    # |xnew - xori| < eps always

    # difference vector
    diff = tf.clip_by_value(tmp-x_ori, -epsilon, epsilon)
    # get point on bound
    tmp= x_ori+diff
    # update of sutable stepsize in the direction
    x_new = x + alpha*(tmp-x)
    x_new = tf.clip_by_value(x_new, 0, 1)
    # will certainly be in the convex set, because alpha < 1 

    x = x_new
    it+=1


  return x, loss, it 






def PGD_targeted(model, x, label,label_name, max_it=10, epsilon=0.05, s=0.03, alpha=1): 
  x_ori = x 
  loss = []
  it = 0

  for it in range(max_it): 
    l, gradient, prediction = util.input_network(model, x, label)
    if (prediction == label_name) :
          break
    loss.append(l)
    tmp = x - s*gradient # TARGETED 

    # "projection"
    # clip because we consider inf norm 
    #tmp = tf.clip_by_value(tmp, -epsilon,epsilon)
    diff = tf.clip_by_value(tmp-x_ori, -epsilon, epsilon)
    # get point on bound
    tmp= x_ori+diff

    x_new = x + alpha*(tmp-x)
    x_new = tf.clip_by_value(x_new, 0, 1)
    # will certainly be in the convex set, because alpha < 1 

    x = x_new
    it+=1


  return x, loss, it 