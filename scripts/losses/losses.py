import tensorflow as tf

def masked_mse_loss(y_true, y_pred, mask=None):
  #mask = y_true != 0
  
  loss = (y_true[1]-y_pred)**2

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss

def mse_cce_loss(y_true, y_pred, alpha=100):
  cce = tf.keras.losses.categorical_crossentropy(y_true[0], y_pred[0])
  mse = masked_mse_loss(y_true[1], y_pred[1])
  return alpha*mse + cce