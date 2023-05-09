import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_MODEL_FEATURES, NUMBER_OF_CLASSES, LIP

class Transformer():
  def __init__(self, 
                model_path, pre_path, target_length,
               last_dense=256, last_dropout=0.2):
    self.model = tf.keras.models.load_model(pre_path + model_path)
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')
    self.dense = tf.keras.layers.Dense(last_dense, activation='relu')
    self.dropout = tf.keras.layers.Dropout(last_dropout)

    self.face_norm =  tf.constant([0.074, 0.07, 0.042])[:target_length]
    self.left_norm =  tf.constant([0.14, 0.22, 0.0383])[:target_length]
    self.pose_norm =  tf.constant([0.27, 0.7, 0.977]  )[:target_length]
    self.right_norm = tf.constant([0.13, 0.15, 0.038] )[:target_length]
    self.norm_concat = tf.keras.layers.Concatenate(axis=2)

  def get_shape(self):
    return (None, 543, 3)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    #face =   self.remove_trend(inputs[:, :, :468, :])  / self.face_norm
    face =   self.remove_trend( tf.gather(inputs, indices=LIP, axis=2) )   / self.face_norm
    left =   self.remove_trend(inputs[:, :, 468:489, :])                   / self.left_norm
    pose =   self.remove_trend(inputs[:, :, 489:522, :])                   / self.pose_norm
    right =  self.remove_trend(inputs[:, :, 522:, :])                      / self.right_norm

    return self.norm_concat([face, left, pose, right])

  def build_model(self):
    inp = tf.keras.layers.Input(self.get_shape())

    
    x1, x2 = self.model(inp)

    pool = self.pooling(x2)
    dense = self.dense(pool)
    dense = self.dropout(dense)
    sign_probs = self.clf_dense(dense)


    out = [x1, sign_probs]
    return tf.keras.models.Model(inp, out)
