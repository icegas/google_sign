import numpy as np
import tensorflow as tf

def masked_mae_loss(y_true, y_pred, mask=None, face_w = 100, hand_w=50):
  #loss = tf.sqrt( (y_true[0]-y_pred[0])**2 )
  rec_loss = tf.abs(y_true[0]-y_pred[0][0]) 
  #mask_loss = tf.keras.backend.binary_crossentropy(target_mask, y_pred[0][1])

  mask = tf.cast(mask, dtype=rec_loss.dtype).numpy()
  div = tf.reduce_sum(mask) 

  mask[:, :, :40, : ] *= face_w
  mask[:, :, 40:61, : ] *= hand_w
  mask[:, :, 94:, : ] *= hand_w
  rec_loss *= mask
  rec_loss = tf.reduce_sum(rec_loss) / div #tf.reduce_sum(mask)

  #y_t_diff = tf.abs( y_true[0][:, 1:] - y_true[0][:, :-1] )
  #y_p_diff = tf.abs( y_pred[0][:, 1:] - y_pred[0][:, :-1] )
  #diff_loss = tf.abs(y_t_diff - y_p_diff)
  #diff_loss *= mask[:, :-1]
  #diff_loss = tf.reduce_sum(diff_loss) / div
  
  return rec_loss, None

def binary_loss(y_true, y_pred, mask=None, face_w=None, hand_w=None):
  mask = mask == -1
  target = tf.reduce_sum( tf.cast( mask, tf.float32), axis=[2, 3])
  target = tf.cast(target!=0, tf.float32)

  loss = tf.keras.losses.binary_crossentropy(target, y_pred[0])
  return tf.reduce_mean(loss)



def mae_cce_loss(y_true, y_pred, mask=None, face_w=100, hand_w=50):
  cce = tf.reduce_mean( tf.keras.losses.categorical_crossentropy(y_true[1], y_pred[1]) )
  mse, _, = masked_mae_loss(y_true, y_pred, mask, face_w=face_w, hand_w=hand_w)
  return mse, cce

def cce_loss(y_true, y_pred, mask=None, face_w=None, hand_w=None):
  return None, tf.reduce_mean( tf.keras.losses.categorical_crossentropy(y_true[1], y_pred[1]) )

class CCE():

  def __init__(self, weights, label_smoothing=0.0, face_w=None, face_h=None) -> None:
    self.loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    self.weights = weights
  
  def calc_loss(self, y_true, y_pred, mask=None):
    
    return None, self.loss(y_true[1], y_pred[1] )

class CCE_MAE():
  def __init__(self, weights, label_smoothing=0.0, face_w=None, hand_w=None) -> None:
    self.cce = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    self.weights = weights
    self.face_w = face_w
    self.hand_w = hand_w
  
  def calc_loss(self, y_true, y_pred, mask=None):
    #mae = masked_mae_loss(y_true, y_pred, mask, self.face_w, self.hand_w)[0]
    mae = binary_loss(y_true, y_pred, mask)
    return mae, self.cce(y_true[1], y_pred[1] )