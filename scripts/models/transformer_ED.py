import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_FEATURES

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               length, target_length, dropout_rate=0.1):
    super().__init__()
    self.spatial_proj_layer = tf.keras.layers.Dense(1)
    self.proj_layer = tf.keras.layers.Dense(d_model)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
      
    self.target_length = target_length
    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES*target_length))
    #self.output_reshape = tf.keras.layers.Reshape((NUMBER_OF_FEATURES, -1, 1))
    self.output_reshape = tf.keras.layers.Reshape((-1, NUMBER_OF_FEATURES, target_length))

    self.final_layer = tf.keras.layers.Dense(NUMBER_OF_FEATURES*target_length)
    #self.final_spatial_proj = tf.keras.layers.Dense(target_length)

    self.concat = tf.keras.layers.Concatenate(axis=2)
    self.face_norm =  tf.convert_to_tensor([0.074, 0.07, 0.042])[:target_length]
    self.left_norm =  tf.convert_to_tensor([0.14, 0.22, 0.0383])[:target_length]
    self.pose_norm =  tf.convert_to_tensor([0.27, 0.7, 0.977]  )[:target_length]
    self.right_norm = tf.convert_to_tensor([0.13, 0.15, 0.038] )[:target_length]
  
  def get_shape(self):
    return (None, 543, 2)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    face =  self.remove_trend(inputs[:, :, :468,  :])  / self.face_norm 
    left =  self.remove_trend(inputs[:, :, 468:489, :]) / self.left_norm
    pose =  self.remove_trend(inputs[:, :, 489:522, :]) / self.pose_norm
    right = self.remove_trend(inputs[:, :, 522:, :]) / self.right_norm

    return self.concat([face, left, pose, right])

  def call(self, inputs):

    normed_input = self.norm_input(inputs)
    #spatial_proj = self.input_reshape( self.spatial_proj_layer(normed_input) )
    spatial_proj = self.input_reshape(normed_input)

    #spatial_proj = self.input_reshape( self.spatial_proj_layer(inputs) )
    proj = self.proj_layer(spatial_proj)

    x = self.encoder(proj) 
    #x = self.decoder(proj, context)

    logits = self.output_reshape( self.final_layer(x) )  

    return logits, normed_input

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
