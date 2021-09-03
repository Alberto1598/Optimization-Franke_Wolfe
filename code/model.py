
class Model :

  def __init__ (self, pretrained_model, decode_predictions, name) :
    
    self.pretrained_model = pretrained_model
    self.decode_predictions = decode_predictions
    self.name = name

  def get_imagenet_label(self, probs):
    return self.decode_predictions(probs, top=1)[0][0]