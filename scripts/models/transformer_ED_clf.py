import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_FEATURES

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               length, target_length, dropout_rate=0.1, cols=2):
    super().__init__()
    self.cols = cols
    self.spatial_proj_layer = tf.keras.layers.Dense(1)
    self.proj_layer = tf.keras.layers.Dense(d_model)

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
    
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_FEATURES, activation='softmax')

    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES))
    #self.output_reshape = tf.keras.layers.Reshape((NUMBER_OF_FEATURES, -1, 1))
    self.output_reshape = tf.keras.layers.Reshape((NUMBER_OF_FEATURES, -1, target_length))

    self.final_layer = tf.keras.layers.Dense(NUMBER_OF_FEATURES*target_length)
    self.final_spatial_proj = tf.keras.layers.Dense(target_length)

    self.concat = tf.keras.layers.Concatenate(axis=1)
    self.bn_face = tf.keras.layers.BatchNormalization()
    self.bn_left = tf.keras.layers.BatchNormalization()
    self.bn_pose = tf.keras.layers.BatchNormalization()
    self.bn_right = tf.keras.layers.BatchNormalization()
  
  def get_shape(self):
    return (None, 543, 2)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    face =  self.bn_face( self.remove_trend(inputs[:, :468, :, :]) )
    left =  self.bn_left( self.remove_trend(inputs[:, 468:489, :, :]) )
    pose =  self.bn_pose( self.remove_trend(inputs[:, 489:522, :, :]) )
    right = self.bn_right( self.remove_trend(inputs[:, 522:, :, :]) )

    return self.concat([face, left, pose, right])

  def call(self, inputs):

    normed_input = self.norm_input(inputs)
    spatial_proj = self.input_reshape( self.spatial_proj_layer(normed_input) )
    #spatial_proj = self.input_reshape( self.spatial_proj_layer(inputs) )
    proj = self.proj_layer(spatial_proj)

    x = self.encoder(proj) 

    logits = self.output_reshape( self.final_layer(x) )  

    pool = self.pooling(x)
    sign_probs = self.clf_dense(pool)
    return logits, sign_probs

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
