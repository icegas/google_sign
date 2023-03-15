import tensorflow as tf
import tensorflow_addons as tfa
from utils.utils import NUMBER_OF_FEATURES

class EncoderConvLayer(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides):
    super().__init__()

    self.conv =  tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', strides=strides)
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

  def call(self, x):
    x = self.relu( self.bn(self.conv(x)) )
    return x

class DecoderConvLayer(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size):
    super().__init__()


    self.up = tf.keras.layers.UpSampling2D()
    self.conv =  tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

  def call(self, x):
    x = self.up(x)
    x = self.relu( self.bn(self.conv(x)) )
    return x

class ConvModel(tf.keras.Model):
  def __init__(self, *, target_length, dropout_rate=0.1):
    super().__init__()

      
    self.target_length = target_length
    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES*target_length, 1))
    self.output_reshape = tf.keras.layers.Reshape((-1, NUMBER_OF_FEATURES, target_length))

    self.conv1_1 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same')
    self.conv1_2 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', dilation_rate=2)
    self.conv1_3 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', dilation_rate=4)
    self.conv_concat = tf.keras.layers.Concatenate()

    self.conv1 = EncoderConvLayer(16, 3, strides=2)
    self.conv2 = EncoderConvLayer(32, 3, strides=2)
    self.conv3 = EncoderConvLayer(64, 3, strides=2)

    self.emb_layer = tf.keras.layers.Dense(544, activation='linear') #136 * 4
    self.reshape_emb_enc = tf.keras.layers.Reshape((-1, 64*136))
    self.reshape_emb_dec = tf.keras.layers.Reshape((-1, 136, 4))
    #self.emb_conv = tf.keras.layers.Conv2D(4, 1, padding='same')

    self.up1 = DecoderConvLayer(64, 3)
    self.up2 = DecoderConvLayer(32, 3)
    self.up3 = DecoderConvLayer(16, 3)

    self.conv =  tf.keras.layers.Conv2D(4, kernel_size=(1, 3), padding='valid')
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

    self.conv_last = tf.keras.layers.Conv2D(1, 1, padding='same')

    self.final_layer = tf.keras.layers.Dense(NUMBER_OF_FEATURES*target_length)

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
    reshape = self.input_reshape(normed_input) 
    
    x1 = self.conv1_1(reshape)
    x2 = self.conv1_2(reshape)
    x3 = self.conv1_3(reshape)
    x = self.conv_concat([x1, x2, x3])

    #x = self.conv1(x)
    #x = self.conv2(x)
    #x = self.conv3(x)
    #emb = self.emb_layer( self.reshape_emb_enc(x) )
    
    #x = self.reshape_emb_dec(emb)
    #x = self.up1(x)
    #x = self.up2(x)
    #x = self.up3(x)
    #x = self.relu(self.bn(self.conv(x)))

    x = self.conv_last(x)

    logits = self.output_reshape( x )  

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
