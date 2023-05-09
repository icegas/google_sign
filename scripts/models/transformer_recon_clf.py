import tensorflow as tf
import tensorflow_addons as tfa
from models.transformer_utils import *
from utils.utils import NUMBER_OF_MODEL_FEATURES, NUMBER_OF_CLASSES, LIP, MASK_VALUE, POSE_IDX, LEFT_HAND_IDX, RIGHT_HAND_IDX

INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform

class Transformer():
  def __init__(self, *, num_layers, d_model, key_dim, num_heads, dff,
               length, target_length, dropout_rate=0.1,
                num_layers_decoder, d_model_decoder, key_dim_decoder,
               dff_decoder,  
               dropout_rate_decoder=0.1, last_dense=256, last_dropout=0.2):
    #self.proj_layer = tf.keras.layers.Dense(d_model, kernel_initializer=INIT_HE_UNIFORM)
    self.proj_layer = tf.keras.layers.Dense(d_model, kernel_initializer=INIT_GLOROT_UNIFORM, activation=tf.keras.activations.gelu)
    self.dec_proj = tf.keras.layers.Dense(d_model, kernel_initializer=INIT_GLOROT_UNIFORM, activation=tf.keras.activations.gelu)
    self.target_length = target_length

    self.encoder = Encoder(num_layers=num_layers, d_model=d_model, key_dim=key_dim,
                           num_heads=num_heads, dff=dff,
                           length=length,
                           dropout_rate=dropout_rate)
    
    self.pooling = tf.keras.layers.GlobalAveragePooling1D()
    self.clf_dense = tf.keras.layers.Dense(NUMBER_OF_CLASSES, activation='softmax', name='outputs', kernel_initializer=INIT_GLOROT_UNIFORM)
    self.dense = tf.keras.layers.Dense(last_dense, activation=tf.keras.activations.gelu, kernel_initializer=INIT_GLOROT_UNIFORM)
    self.dropout = tf.keras.layers.Dropout(last_dropout)

    self.input_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES * target_length))

    #self.face_embedding  = tf.keras.layers.Dense(64)
    #self.left_embedding  = tf.keras.layers.Dense(64)
    #self.pose_embedding  = tf.keras.layers.Dense(64)
    #self.right_embedding = tf.keras.layers.Dense(64)
    n_units = 128

    self.face_embedding  = LandmarkEmbedding(256, n_units, len(LIP) * target_length)
    self.left_embedding  = LandmarkEmbedding(256, n_units, 21 * target_length)
    self.pose_embedding  = LandmarkEmbedding(256, n_units, 33 * target_length)
    self.right_embedding = LandmarkEmbedding(256, n_units, 21 * target_length)

    self.concat = tf.keras.layers.Concatenate(axis=-1)
    self.face_mean =   tf.constant( [0.4658, 0.4702, -0.02327] )[:self.target_length]
    self.left_mean =   tf.constant( [0.6113, 0.573, -0.05847]  )[:self.target_length]
    self.pose_mean =   tf.constant( [0.4949, 1.045, -0.68]     )[:self.target_length]
    self.right_mean = tf.constant( [0.3384, 0.573, -0.05847]   )[:self.target_length]

    self.face_std = tf.constant([0.06, 0.0535, 0.0145]          )[:self.target_length]
    self.left_std = tf.constant([0.1215666, 0.157338, 0.058522] )[:self.target_length]
    self.pose_std = tf.constant([0.27467, 0.694649, 0.970158748])[:self.target_length]
    self.right_std = tf.constant([0.1215666, 0.157338, 0.058522])[:self.target_length]

    self.norm_concat = tf.keras.layers.Concatenate(axis=2)
    self.length = length
    self.positional_embedding = tf.keras.layers.Embedding(length+1, n_units*4, embeddings_initializer=tf.keras.initializers.constant(0.0))

    self.decoder = Encoder(num_layers=num_layers_decoder, d_model=d_model_decoder,
                           key_dim=key_dim_decoder, dff=dff_decoder,
                           length=length, dropout_rate=dropout_rate_decoder, num_heads=num_heads)

    self.decoder_proj_mask = tf.keras.layers.Dense(NUMBER_OF_MODEL_FEATURES * target_length)
    self.decoder_proj = tf.keras.layers.Dense(NUMBER_OF_MODEL_FEATURES * target_length)
    self.output_reshape = tf.keras.layers.Reshape((-1,  NUMBER_OF_MODEL_FEATURES, target_length))
  
  def get_shape(self):
    return (None, 543, 3)


  def get_embedding(self, inputs):

    face =   self.face_embedding( inputs[:, :, :40, :] )   
    left =   self.left_embedding( inputs[:, :, 40:61, :] ) 
    pose =   self.pose_embedding( inputs[:, :, 61:94, :] ) 
    right = self.right_embedding(  inputs[:, :, 94:, :] )  

    return self.concat([face, left, pose, right])   

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
  
  def norm_input(self, inputs):

    face =  tf.gather(inputs, indices=LIP, axis=2)[:, :, :, :self.target_length]
    mask = tf.math.not_equal(face, 0)#face != 0
    face = (face - self.face_mean ) / self.face_std
    face = tf.where(mask, face, tf.zeros_like(face) )

    mask = tf.math.not_equal( inputs[:, :, LEFT_HAND_IDX, :self.target_length], 0 )
    left = ( inputs[:, :, LEFT_HAND_IDX, :self.target_length] - self.left_mean ) / self.left_std
    left = tf.where(mask, left, tf.zeros_like(left) )

    mask = tf.math.not_equal( inputs[:, :, RIGHT_HAND_IDX, :self.target_length], 0 )
    right = (inputs[:, :, RIGHT_HAND_IDX, :self.target_length] - self.right_mean ) / self.right_std
    right = tf.where(mask, right, tf.zeros_like(right) )

    pose = (inputs[:, :, POSE_IDX, :self.target_length] - self.pose_mean) / self.pose_std


    x = tf.concat([
            face, left, pose, right
        ],2 )

    return x

  def build_model(self):
    inp = tf.keras.layers.Input(self.get_shape())


    norm_inp = self.norm_input(inp) 
    normalised_non_empty_frame_idxs = self.get_positional_embedding(norm_inp)
    embs = self.get_embedding(norm_inp) #+ self.positional_embedding(normalised_non_empty_frame_idxs)
    dec = self.decoder(self.dec_proj(embs))
    reconstruction = self.output_reshape( self.decoder_proj(dec) )
    
    embs = self.get_embedding(reconstruction) + self.positional_embedding(normalised_non_empty_frame_idxs)
    proj = self.proj_layer(embs)

    x = self.encoder(proj) 


    pool = self.pooling(x)
    dense = self.dense(pool)
    dense = self.dropout(dense)
    sign_probs = self.clf_dense(dense)

    out = [reconstruction, sign_probs]
    return tf.keras.models.Model(inp, out)
