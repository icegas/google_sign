import tensorflow as tf
import tensorflow_addons as tfa
from utils.utils import NUMBER_OF_FEATURES, LIP, NUMBER_OF_CLASSES, NUMBER_OF_MODEL_FEATURES, MASK_VALUE

class EncoderConvLayer(tf.keras.layers.Layer):
  def __init__(self, filters, kernel_size, strides, convs):
    super().__init__()

    self.convs, self.bns, self.relus = [], [], []
    for i in range(convs):
      self.convs.append( tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, padding='same', dilation_rate=i+1, strides=strides) )
      self.bns.append( tf.keras.layers.BatchNormalization() )
      self.relus.append( tf.keras.layers.ReLU() )

    self.concat = tf.keras.layers.Add()
    self.mp = tf.keras.layers.MaxPool2D()

  def call(self, x):
    
    outs = []
    for i in range(len(self.convs)):
      outs.append(  self.relus[i]( self.bns[i](self.convs[i](x)) ) )
    outs = self.concat(outs)

    return self.mp( outs )

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

class ConvModel():
  def __init__(self, *, target_length, dropout_rate=0.1):
    #super().__init__()

      
    self.target_length = target_length
    #self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_FEATURES*target_length, 1))
    #self.output_reshape = tf.keras.layers.Reshape((-1, NUMBER_OF_FEATURES, target_length))

    self.conv1_1 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same')
    self.conv1_2 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', dilation_rate=2)
    self.conv1_3 = tf.keras.layers.Conv2D(16, kernel_size=5, padding='same', dilation_rate=4)
    self.conv_concat = tf.keras.layers.Concatenate()

    self.flatten = tf.keras.layers.Flatten()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')
    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES * target_length, 1))

    self.pad = tf.keras.layers.ZeroPadding2D(padding=( (0, 0), (0, 13)) )
    self.conv1 = EncoderConvLayer(32, 3, strides=1, convs=3)
    self.conv2 = EncoderConvLayer(64, 3, strides=1, convs=3)
    self.conv3 = EncoderConvLayer(128, 3, strides=1, convs=3)

    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')

    #self.emb_conv = tf.keras.layers.Conv2D(4, 1, padding='same')
    self.conv_x =  tf.keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same')
    self.bn_x = tf.keras.layers.BatchNormalization()
    self.relu_x = tf.keras.layers.ReLU()

    self.up1 = DecoderConvLayer(64, 3) #conv2
    self.up2 = DecoderConvLayer(32, 3) #conv1
    self.up3 = DecoderConvLayer(3, 3)

    self.conv =  tf.keras.layers.Conv2D(4, kernel_size=(3, 3), padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

    self.conv_last = tf.keras.layers.Conv2D(3, 1, padding='same')


    self.concat = tf.keras.layers.Concatenate(axis=2)
    self.output_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES, target_length))
    self.crop = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 13)) )#352

  
  def get_shape(self):
    return (None, 543, 3)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    
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

  def build_model(self):

    inp = tf.keras.layers.Input(self.get_shape())
    normed_input =  self.norm_input(inp) 
    normed_input = self.pad(normed_input) 
    x1 = self.conv1(normed_input)
    x2 = self.conv2(x1)
    x3 = self.conv3(x2)
    x4 = self.relu_x(self.bn_x(self.conv_x(x3))) 

    
    up1 = self.up1(x4)  + x2
    up2 = self.up2(up1) + x1
    up3 = self.up3(up2) + normed_input
    x = self.crop(up3)
    x = self.relu(self.bn(self.conv(x)))

    logits = self.conv_last(x)
    #logits = self.output_reshape( x )  
    out = [logits, x]
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
