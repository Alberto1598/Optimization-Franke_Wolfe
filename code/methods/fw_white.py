import tensorflow as tf
import util


def FW_white_untargeted(model, x, label, label_name, max_it = 10, epsilon = 0.05, step_size = 0.03,  beta=0.9): 
  l, m, _ = util.input_network(model,x, label) 
  m = -m
  x_ori = x
  loss = []
  it = 0

  for it in range(max_it):
    l, grad, prediction = util.input_network(model, x, label) 
    if (prediction != label_name):
          break
    #m_new = beta * m + (1-beta)* grad 
    # forse bisogna cambiare solo qui il segno e non sotto, 
    # altriment si cambia tutto m e non solo il gradiente 
    m_new = beta * m - (1-beta)* grad 
    
    loss.append(l)

    v = - epsilon*tf.sign(m_new) + x_ori # UNTARGETED 
    # we need + and not -, because we need the gradient ascent 
    d = v - x 
    x_new = x + step_size*d 
    x_new = tf.clip_by_value(x_new, 0, 1)

    x = x_new 
    m = m_new
    it += 1 
  
  return x, loss, it 


def FW_white_targeted(model, x, label,label_name, max_it, epsilon, step_size, beta=0.9): 
  l, m, _ = util.input_network(model,x, label) 
  x_ori = x
  loss = []
  it = 0
  for it in range(max_it):
    l, grad, prediction = util.input_network(model, x,label)
    if (prediction == label_name):
          break 
    m_new = beta * m + (1-beta)* grad 
    loss.append(l)

    v = - epsilon*tf.sign(m_new) + x_ori # TARGETED 
    # we need + and not -, because we need the gradient ascent 
    d = v - x 
    x_new = x + step_size*d 
    x_new = tf.clip_by_value(x_new, 0, 1)

    x = x_new 
    m = m_new 
    it += 1
  
  return x, loss, it


