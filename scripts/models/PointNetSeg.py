import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
import sys
sys.path.append('/home/icegas/Desktop/kaggle/google_sign/google_sign/scripts/')
from utils.utils import MASK_VALUE, LIP

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    """Reference: https://keras.io/examples/vision/pointnet/#build-a-model"""

    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.identity = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.identity))

    def get_config(self):
        config = super().get_config()
        config.update({"num_features": self.num_features, "l2reg_strength": self.l2reg})
        return config

def conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Conv2D(filters, kernel_size=1, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def mlp_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    x = layers.Dense(filters, name=f"{name}_dense")(x)
    x = layers.BatchNormalization(momentum=0.0, name=f"{name}_batch_norm")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)

def transformation_net(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    """
    Reference: https://keras.io/examples/vision/pointnet/#build-a-model.

    The `filters` values come from the original paper:
    https://arxiv.org/abs/1612.00593.
    """
    x = conv_block(inputs, filters=16, name=f"{name}_1")
    x = conv_block(x, filters=32, name=f"{name}_2")
    x = conv_block(x, filters=128, name=f"{name}_3")
    x = tf.reduce_max(x, axis=2)
    x = mlp_block(x, filters=64, name=f"{name}_1_1")
    x = mlp_block(x, filters=32, name=f"{name}_2_1")
    return layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(np.eye(num_features).flatten()),
        activity_regularizer=OrthogonalRegularizer(num_features),
        name=f"{name}_final",
    )(x)


def transformation_block(inputs: tf.Tensor, num_features: int, name: str) -> tf.Tensor:
    transformed_features = transformation_net(inputs, num_features, name=name)
    transformed_features = layers.Reshape((-1, num_features, num_features))(
        transformed_features
    )
    return layers.Dot(axes=( 3, 2), name=f"{name}_mm")([inputs, transformed_features])

def get_shape_segmentation_model(input_points, num_points: int, num_classes: int) -> keras.Model:
    #input_points = keras.Input(shape=(None, None, 3))

    # PointNet Classification Network.
    features_64 = conv_block(input_points, filters=16, name="features_64")
    features_128_1 = conv_block(features_64, filters=32, name="features_128_1")
    features_128_2 = conv_block(features_128_1, filters=32, name="features_128_2")
    features_512 = conv_block(features_128_2, filters=256, name="features_512")
    features_2048 = conv_block(features_512, filters=512, name="pre_maxpool_block")
    global_features = tf.reduce_max(features_2048, axis=2, keepdims=True)
    
    #global_features = tf.expand_dims(global_features, axis=[2])
    #global_features = tf.tile(global_features, [1, 1, num_points, 1])
    global_features = tf.repeat(global_features, num_points, axis=2)

    # Segmentation head.
    segmentation_input = layers.Concatenate(name="segmentation_input")(
        [
            features_64,
            features_128_1,
            features_128_2,
            features_512,
            global_features,
        ]
    )
    segmentation_features = conv_block(
        segmentation_input, filters=128, name="segmentation_features"
    )
    outputs = layers.Conv2D(
        num_classes, kernel_size=1, activation="linear", name="segmentation_head"
    )(segmentation_features)
    return keras.Model(input_points, outputs)

class PointNet():

    def __init__(self, num_points, num_classes) -> None:
        self.num_points = num_points
        self.num_classes = num_classes

    def norm_input(self, inputs):
        #face =   self.remove_trend(inputs[:, :, :468, :])  / self.face_norm
    
        mask_0 = inputs != 0 
        mask =  inputs != MASK_VALUE
        #inputs = inputs - inputs[:, :, 489:490, :]

        mean_mask = mask_0 & mask

        inputs -= tf.reduce_sum(inputs, axis=[1, 2], keepdims=True) / tf.reduce_sum( 
          tf.cast(mean_mask, tf.float32), axis=[1, 2], keepdims=True)
        inputs /= tf.reduce_max(tf.norm(inputs, axis=-1, keepdims=True), axis=1, keepdims=True)

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
        input_points = keras.Input(shape=(None, None, 3))
        norm_inp = self.norm_input(input_points)
        return get_shape_segmentation_model(norm_inp, self.num_points, self.num_classes)
    

if __name__ == '__main__':
    model = PointNet(543, 3).build_model()
    inp = np.random.normal(size=(64, 128, 543, 3))

    out = model(inp)
    print(out.shape) 
