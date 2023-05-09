import tensorflow as tf
import tensorflow_addons as tfa
from utils.utils import NUMBER_OF_FEATURES, LIP, NUMBER_OF_CLASSES, NUMBER_OF_MODEL_FEATURES, MASK_VALUE

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

    self.pad = tf.keras.layers.ZeroPadding((0, 0), (0, 13))
    self.conv1 = EncoderConvLayer(16, 5, strides=2)
    self.conv2 = EncoderConvLayer(32, 5, strides=2)
    self.conv3 = EncoderConvLayer(64, 5, strides=2)


    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs')

    #self.emb_conv = tf.keras.layers.Conv2D(4, 1, padding='same')

    self.up1 = DecoderConvLayer(64, 3)
    self.up2 = DecoderConvLayer(32, 3)
    self.up3 = DecoderConvLayer(16, 3)

    self.conv =  tf.keras.layers.Conv2D(4, kernel_size=(3, 3), padding='same')
    self.bn = tf.keras.layers.BatchNormalization()
    self.relu = tf.keras.layers.ReLU()

    self.conv_last = tf.keras.layers.Conv2D(1, 1, padding='same')


    self.dense = tf.keras.layers.Dense(512, activation='relu')
    self.dropout = tf.keras.layers.Dropout(0.4)
    self.final_layer = tf.keras.layers.Dense(NUMBER_OF_FEATURES*target_length)

    self.concat = tf.keras.layers.Concatenate(axis=2)
    self.output_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES, target_length))
    self.crop = tf.keras.layers.Cropping2D(cropping=((0, 0), (0, 7)) )#352

  
  def get_shape(self):
    return (None, 543, 3)

  def remove_trend(self, x):
    return tf.subtract(x,  tf.reduce_mean(x, axis=(2), keepdims=True) )

  def norm_input(self, inputs):
    #face =   self.remove_trend(inputs[:, :, :468, :])  / self.face_norm
    
    mask_0 = inputs != 0 
    mask =  inputs != MASK_VALUE
    inputs = inputs - inputs[:, :, 489:490, :]
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
    x = self.conv1(normed_input)
    x = self.conv2(x)
    x = self.conv3(x)
    pool = tf.reduce_mean(x, axis=1)
    pool = self.flatten(pool)

    pool = self.dense(pool)
    pool = self.dropout(pool) 
    sign_probs = self.clf_dense(pool)
    
    #x = self.reshape_emb_dec(emb)
    x = self.up1(x)
    x = self.up2(x)
    x = self.up3(x)
    x = self.relu(self.bn(self.conv(x)))

    x = self.conv_last(x)
    logits = self.output_reshape( x )  
    out = [logits, sign_probs]
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
