import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_MODEL_FEATURES, NUMBER_OF_CLASSES, LIP, LEFT_HAND_IDX, RIGHT_HAND_IDX, MASK_VALUE
from utils.utils import NUMBER_OF_FEATURES

class Transformer():
  def __init__(self, *, num_layers, d_model, key_dim, num_heads, dff,
               length, target_length, num_layers_decoder, d_model_decoder, key_dim_decoder,
               dff_decoder, 
               dropout_rate=0.1, 
               dropout_rate_decoder=0.1, last_dense=256, last_dropout=0.2):
    self.proj_layer = tf.keras.layers.Dense(d_model)
    #self.proj_layer_decoder = tf.keras.layers.Dense(d_model_decoder)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, key_dim=key_dim,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
    
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')
    self.dense = tf.keras.layers.Dense(last_dense, activation='relu')
    self.dropout = tf.keras.layers.Dropout(last_dropout)

    #self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES * target_length))
    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES * target_length))

    self.face_embedding  = tf.keras.layers.Dense(64)
    self.left_embedding  = tf.keras.layers.Dense(64)
    self.pose_embedding  = tf.keras.layers.Dense(64)
    self.right_embedding = tf.keras.layers.Dense(64)

    self.face_reshape = tf.keras.layers.Reshape((-1,  len(LIP) * target_length))
    self.left_reshape = tf.keras.layers.Reshape((-1,  (489-468) * target_length))
    self.pose_reshape = tf.keras.layers.Reshape((-1,  (522-489) * target_length))
    self.right_reshape = tf.keras.layers.Reshape((-1, (543-522) * target_length))

    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.face_norm =  tf.constant([0.074, 0.07, 0.042])[:target_length]
    self.left_norm =  tf.constant([0.14, 0.22, 0.0383])[:target_length]
    self.pose_norm =  tf.constant([0.27, 0.7, 0.977]  )[:target_length]
    self.right_norm = tf.constant([0.13, 0.15, 0.038] )[:target_length]


    self.decoder = Encoder(num_layers=num_layers_decoder, d_model=d_model_decoder,
                           key_dim=key_dim_decoder, dff=dff_decoder,
                           length=length, dropout_rate=dropout_rate_decoder, num_heads=num_heads)

    self.decoder_proj = tf.keras.layers.Dense(NUMBER_OF_MODEL_FEATURES * target_length)
    self.output_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES, target_length))

    self.norm_concat = tf.keras.layers.Concatenate(axis=2)
    self.length = tf.constant(length)
    self.target_length = target_length
  
  def get_shape(self):
    return (None, 543, 3)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    return inputs
    ##face =   self.remove_trend(inputs[:, :, :468, :])  / self.face_norm
    #face =   self.remove_trend( tf.gather(inputs, indices=LIP, axis=2)[:, :, :, :self.target_length] )   / self.face_norm
    #left =   self.remove_trend(inputs[:, :, 468:489, :self.target_length])                   / self.left_norm
    #pose =   self.remove_trend(inputs[:, :, 489:522, :self.target_length])                   / self.pose_norm
    #right =  self.remove_trend(inputs[:, :, 522:,    :self.target_length])                      / self.right_norm

    #return self.norm_concat([face, left, pose, right])

  def get_poses(self, inputs):
    
    return self.norm_concat([tf.gather(inputs, indices=LIP, axis=2), 
        inputs[:, :, 468:489, :], inputs[:, :, 489:522, :], inputs[:, :, 522:, :]])
  
  #def mirror_hand(self, input):

  #  #index = (input[:, LEFT_HAND_IDX] == MASK_VALUE).sum()
  #  mirror_inp = tf.scalar_mul(-1.0, input)

  #  left_inp = tf.gather(mirror_inp, indices=range(468,489), axis=2)
  #  right_inp = tf.gather(mirror_inp, indices=range(522,543), axis=2)
  #  tf.tensor_scatter_nd_update( mirror_inp, indices=range(468, 489), updates=right_inp)
  #  nan_left = tf.reduce_sum(left_inp == -3, axis=[-1, -2])
  #  nan_right = tf.reduce_sum(left_inp == -3, axis=[-1, -2] )

  #  return tf.where(nan_left > nan_right, input, mirror_inp)

  def get_embedding(self, inputs):

    #inputs = self.mirror_hand(inputs)
    face =   self.face_embedding(  ( self.face_reshape( 
       self.remove_trend(  tf.gather(inputs, indices=LIP, axis=2)[:, :, :, :self.target_length] )   / self.face_norm ) ) )
    left =   self.left_embedding(  ( self.left_reshape(  self.remove_trend( inputs[:, :, 468:489, :self.target_length] )                  / self.left_norm ) ) )
    pose =   self.pose_embedding(  ( self.pose_reshape(  self.remove_trend( inputs[:, :, 489:522, :self.target_length] )                  / self.pose_norm ) ) )
    right =  self.right_embedding( ( self.right_reshape( self.remove_trend( inputs[:, :, 522:,    :self.target_length] )                     / self.right_norm) ) ) 

    #inputs = self.remove_trend(inputs)
    #face =   self.face_embedding(   self.face_reshape( tf.gather(inputs, indices=LIP, axis=2) ) ) 
    #left =   self.left_embedding(   self.left_reshape(inputs[:, :, 468:489, :] ) ) 
    #pose =   self.pose_embedding(   self.pose_reshape(inputs[:, :, 489:522, :] ) ) 
    #right =  self.right_embedding(  self.right_reshape(inputs[:, :, 522:, :] ) ) 

    return self.concat([face, left, pose, right])

  def get_embedding_2(self, inputs):

    face =   self.face_embedding(  ( self.face_reshape(   inputs[:, :, :40, :]) ) )  # / self.face_norm )
    left =   self.left_embedding(  ( self.left_reshape(   inputs[:, :, 40:61, :]  ) ) )  #/ self.left_norm )
    pose =   self.pose_embedding(  ( self.pose_reshape(   inputs[:, :, 61:94, :]  ) ) )  #/ self.pose_norm )
    right =  self.right_embedding( ( self.right_reshape(  inputs[:, :, 94:, :] )  ) )  #/ self.right_norm )

    #inputs = self.remove_trend(inputs)
    #face =   self.face_embedding(   self.face_reshape( tf.gather(inputs, indices=LIP, axis=2) ) ) 
    #left =   self.left_embedding(   self.left_reshape(inputs[:, :, 468:489, :] ) ) 
    #pose =   self.pose_embedding(   self.pose_reshape(inputs[:, :, 489:522, :] ) ) 
    #right =  self.right_embedding(  self.right_reshape(inputs[:, :, 522:, :] ) ) 

    return self.concat([face, left, pose, right])

  #def call(self, inputs):

  #  embs = self.get_embedding(inputs)
  #  #spatial_proj = self.bn( self.input_reshape( inputs ) )
  #  proj = self.proj_layer(embs)

  #  x = self.encoder(proj) 
  #  pool = self.pooling(x)
  #  sign_probs = self.clf_dense(pool)

  #  reconstruction = self.decoder(x)
  #  reconstruction = self.decoder_proj(reconstruction)

  #  return reconstruction, sign_probs

  def build_model(self):
    inp = tf.keras.layers.Input(self.get_shape())

    pad = tf.pad(inp,  [[0, 0], [0, tf.reduce_max([0, self.length-tf.shape(inp)[1]])], [0, 0], [ 0, 0]]  , 'CONSTANT')
    pad = self.input_reshape(pad)

    #embs = self.get_embedding(pad)
    proj = self.proj_layer(pad)

    dec = self.decoder(proj)
    reconstruction = self.output_reshape( self.decoder_proj(dec) )

    embs_2 = self.get_embedding(reconstruction)
    proj_2 = self.proj_layer(embs_2)

    #x = self.encoder(dec) 
    x = self.encoder(proj_2) 
    pool = self.pooling(x)
    dense = self.dense(pool)
    dense = self.dropout(dense)
    sign_probs = self.clf_dense(dense)


    out = [reconstruction, sign_probs]
    return tf.keras.models.Model(inp, out)

#if __name__=='__main__':
#  model = Transformer(num_layers=2, d_model=256, num_heads=8, dff=512, length=400, target_length=64)
#
#
#  inp = tf.convert_to_tensor( np.random.normal(size=[4, 128, 64]) )
#  output = model(inp, training=False)
#  print(output.shape)
#
#  inp = tf.convert_to_tensor( np.random.normal(size=[4, 256, 64]) )
#  output = model(inp, training=False)
#  print(output.shape)
