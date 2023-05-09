import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_MODEL_FEATURES, NUMBER_OF_CLASSES, LIP, MASK_VALUE

class Transformer():
  def __init__(self, *, num_layers, d_model, key_dim, num_heads, dff,
               length, target_length, dropout_rate=0.1, 
               dropout_rate_decoder=0.1, last_dense=256, last_dropout=0.2):
    self.proj_layer = tf.keras.layers.Dense(d_model)
    self.target_length = target_length

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, key_dim=key_dim,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
    
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')
    self.dense = tf.keras.layers.Dense(last_dense, activation='relu')
    self.dropout = tf.keras.layers.Dropout(last_dropout)

    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES * target_length))

    #self.face_embedding  = tf.keras.layers.Dense(64)
    #self.left_embedding  = tf.keras.layers.Dense(64)
    #self.pose_embedding  = tf.keras.layers.Dense(64)
    #self.right_embedding = tf.keras.layers.Dense(64)

    self.face_embedding  = LandmarkEmbedding(256, 64, len(LIP) * target_length)
    self.left_embedding  = LandmarkEmbedding(256, 64, 21 * target_length)
    self.pose_embedding  = LandmarkEmbedding(256, 64, 33 * target_length)
    self.right_embedding = LandmarkEmbedding(256, 64, 21 * target_length)

    self.concat = tf.keras.layers.Concatenate(axis=1)
    self.face_norm =  tf.constant([0.074, 0.07, 0.042])[:target_length]
    self.left_norm =  tf.constant([0.14, 0.22, 0.0383])[:target_length]
    self.pose_norm =  tf.constant([0.27, 0.7, 0.977]  )[:target_length]
    self.right_norm = tf.constant([0.13, 0.15, 0.038] )[:target_length]


    self.length = tf.constant(length)
  
  def get_shape(self):
    return (None, 543, 3)


  def norm_input(self, inputs):
    #face =   self.remove_trend(inputs[:, :, :468, :])  / self.face_norm
    
    mask_0 = inputs != 0 
    mask =  inputs != MASK_VALUE
    #inputs = inputs - inputs[:, :, 489:490, :]

    mean_mask = mask_0 & mask

    inputs -= tf.reduce_sum(inputs, axis=[1, 2], keepdims=True) / tf.reduce_sum( 
      tf.cast(mean_mask, tf.float32), axis=[1, 2], keepdims=True)
    inputs /= tf.reduce_max(tf.norm(inputs, axis=-1, keepdims=True), axis=[1, 2], keepdims=True )

    #x -= tf.reduce_sum(x, axis=[1, 2], keepdims=True) / tf.reduce_sum( 
    #tf.cast(mask, tf.float32), axis=[1, 2], keepdims=True)
    #x /= tf.reduce_max(tf.norm(x, axis=-1, keepdims=True), axis=[1, 2], keepdims=True)

    inputs = tf.where(mask, inputs, tf.zeros_like(inputs) + MASK_VALUE)
    inputs = tf.where(mask_0, inputs, tf.zeros_like(inputs) )

    x = tf.concat([
            tf.gather(inputs, indices=LIP, axis=2),
            inputs[:, :, 468:489],
            inputs[:, :, 489:522],
            inputs[:, :, 522:]
        ],2 )

    return x

  
  def get_embedding(self, inputs):

    face =   self.face_embedding( inputs[:, :, :40, :] ) 
    left =   self.left_embedding( inputs[:, :, 40:61, :] )  
    pose =   self.pose_embedding( inputs[:, :, 61:94, :] ) 
    right = self.right_embedding(  inputs[:, :, 94:, :] ) 

    return self.concat([face, left, pose, right])

  def build_model(self):
    inp = tf.keras.layers.Input(self.get_shape())

    norm_inp = self.norm_input(inp)
    #pad = tf.pad(inp,  [[0, 0], [0, tf.reduce_max([0, self.length-tf.shape(inp)[1]])], [0, 0], [ 0, 0]]  , 'CONSTANT')

    embs = self.get_embedding(norm_inp)
    proj = self.proj_layer(embs)

    x = self.encoder(proj) 
    pool = self.pooling(x)
    dense = self.dense(pool)
    dense = self.dropout(dense)
    sign_probs = self.clf_dense(dense)


    out = [proj, sign_probs]
    return tf.keras.models.Model(inp, out)
