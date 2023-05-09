import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_MODEL_FEATURES, NUMBER_OF_CLASSES, LIP, MASK_VALUE, POSE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX
from utils.utils import LEFT_POSE_IDXS, RIGHT_POSE_IDXS

INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform

class Transformer():
  def __init__(self, *, num_layers, d_model, key_dim, num_heads, dff,
               length, target_length, dropout_rate=0.1, 
               dropout_rate_decoder=0.1, last_dense=256, last_dropout=0.2, pooling='SAP', heads=None, hand_model_path=None):
    
    self.proj_layer = tf.keras.layers.Dense(d_model, kernel_initializer=INIT_GLOROT_UNIFORM, activation=tf.keras.activations.gelu)
    self.target_length = target_length
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, key_dim=key_dim,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
    
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs', kernel_initializer=INIT_GLOROT_UNIFORM)
    self.dense = tf.keras.layers.Dense(last_dense, activation=tf.keras.activations.gelu, kernel_initializer=INIT_GLOROT_UNIFORM)
    self.dropout = tf.keras.layers.Dropout(last_dropout)

    #12
    #96 + 160 + 256
    n_units = 96 + 160 + 256
    features = 9
    self.face_embedding  = LandmarkEmbedding(256, 160, len(LIP) * features, 128 )
    self.hand_embedding  = LandmarkEmbedding(256, 256, 21 * features, 128 )
    self.pose_embedding  = LandmarkEmbedding(256, 96, len(LEFT_POSE_IDXS) *  features, 16 )
    #self.right_embedding = LandmarkEmbedding(256, n_units, 21 *       (9) )

    self.length = length
    self.positional_embedding = tf.keras.layers.Embedding(self.length+1, n_units, embeddings_initializer=tf.keras.initializers.constant(0.0))
    self.set_pooling(pooling, d_model, heads)

    self.feature = FeatureLayer()
    #self.landmark_weights = tf.Variable(tf.zeros([3], dtype=tf.float32) )
  
  def set_pooling(self, pooling, d_model, heads):
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    if pooling == 'SAP':
      self.pooling = SAP(d_model, 1)
    elif pooling == 'ASP':
      self.pooling = ASP(d_model, 1)
    elif pooling == 'MSAP':
      self.pooling = MSAP(d_model, heads, 1)
  def get_shape(self):
    return (None, 543, 3)

  def norm_input(self, inputs):

    mask = tf.math.not_equal(inputs, 0)
    inputs = inputs - inputs[:, :1, 489:490, :]
    inputs = tf.where(mask, inputs, tf.zeros_like(inputs) )

    face =  tf.gather(inputs, indices=LIP, axis=2)[:, :, :, :self.target_length]
    left =  inputs[:, :, LEFT_HAND_IDX, :self.target_length]
    pose_left = tf.gather(inputs, indices=LEFT_POSE_IDXS, axis=2)[:, :, :, :self.target_length]
    right = inputs[:, :, RIGHT_HAND_IDX, :self.target_length]
    right = tf.concat(
      [tf.multiply( tf.gather(right, [0], axis=-1), -1 ), tf.gather(right, [1], axis=-1) ], axis=-1 
    )
    pose_right = tf.gather(inputs, indices=RIGHT_POSE_IDXS, axis=2)[:, :, :, :self.target_length] 
    pose_right = tf.concat(
      [tf.multiply( tf.gather(pose_right, [0], axis=-1), -1 ), tf.gather(pose_right, [1], axis=-1) ], axis=-1 
    )
    face_right = tf.concat(
      [tf.multiply( tf.gather(face, [0], axis=-1), -1 ), tf.gather(face, [1], axis=-1) ], axis=-1 
    )

    right_mask = tf.reduce_sum( tf.cast( tf.math.equal(right, 0), tf.float32), axis=[1, 2, 3] )
    left_mask =  tf.reduce_sum( tf.cast( tf.math.equal(left, 0),  tf.float32), axis=[1, 2, 3] )



    #hand rec - reconstruction hand
    left_all =  tf.concat([face, pose_left,          left  ], 2)
    right_all = tf.concat([face_right, pose_right, right ], 2)
    w =  tf.cast( tf.math.greater( right_mask, left_mask ), tf.float32 )
    for i in range(3):
      w = tf.expand_dims(w, axis=-1)

    mask = tf.multiply(w, (tf.zeros_like(right_all, dtype=tf.float32) + 1) )
    x = tf.where(tf.cast(mask, tf.bool), left_all, right_all)
    #x = tf.concat([x, hand_rec], 2)

    x = self.feature(x)

    return x

  
  def get_embedding(self, inputs):

    face =   self.face_embedding( inputs[:, :, :len(LIP), :] )   
    pose =   self.pose_embedding( inputs[:, :, len(LIP):len(LIP) + len(LEFT_POSE_IDXS):, :] ) 
    hand =   self.hand_embedding( inputs[:, :, len(LIP)+len(LEFT_POSE_IDXS):, :])
    x = tf.concat([face, pose, hand], 2)
    return x

  
  def get_positional_embedding(self, inp):
    mask = tf.cast( tf.math.not_equal(inp, 0), tf.int32)
    out = tf.reduce_sum(mask, axis=[2, 3])

    mask = tf.cast( tf.math.not_equal(out, 0), tf.int32)
    mask_0 = tf.math.not_equal( mask, 0 )

    mask = tf.where( mask_0, tf.cumsum(mask, axis=1), 0 )
    non_empty_frame_idxs = tf.cast(mask, tf.float32)

    max_frame_idxs = tf.clip_by_value(
                tf.reduce_max(non_empty_frame_idxs, axis=1, keepdims=True),
                1,
                np.PINF,
            )

    normalised_non_empty_frame_idxs = tf.where(
            tf.math.equal(non_empty_frame_idxs, 0.0),
            self.length,
            tf.cast(
                non_empty_frame_idxs / (max_frame_idxs + 1) * self.length,
                tf.int32,
            ),
        )
    
    return normalised_non_empty_frame_idxs

  def build_model(self):
    inp = tf.keras.layers.Input(self.get_shape())
    mask = tf.cast( tf.math.not_equal(inp, 0), tf.float32)
    mask = tf.reduce_sum(mask, axis=[2, 3])

    #hand_inp = tf.where(tf.math.equal(inp, 0), 1.0, inp)
    #hand = self.hand_model(hand_inp) * tf.expand_dims( 
    # tf.expand_dims( tf.cast( tf.math.not_equal(mask, 0), tf.float32) , -1), -1 )
    #hand = tf.stop_gradient(hand)

    mask = tf.cast( tf.math.equal(mask, 0), tf.float32) * -1e9
    norm_inp = self.norm_input(inp) 
    
    normalised_non_empty_frame_idxs = self.get_positional_embedding(inp)
    #proj = self.get_embedding(norm_inp) + self.positional_embedding(normalised_non_empty_frame_idxs)
    proj =  tf.concat( [self.get_embedding(norm_inp),
                       self.positional_embedding(normalised_non_empty_frame_idxs)],
                       -1 )
    proj = self.proj_layer(proj) 

    x = self.encoder(proj) 
    pool = self.pooling(x, mask)
    dense = self.dense(pool)
    dense = self.dropout(dense)
    sign_probs = self.clf_dense(dense)

    out = [norm_inp, sign_probs]
    return tf.keras.models.Model(inp, out)
