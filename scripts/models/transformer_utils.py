import numpy as np
import tensorflow as tf
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Layer, Dense
from keras.backend import softmax

INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform

class FeatureLayer(Layer):
    def __init__(self, **kwargs) -> None:
      super(FeatureLayer, self).__init__(**kwargs)
      self.bn = tf.keras.layers.BatchNormalization()

    def get_config(self):
      base_config = super(FeatureLayer, self).get_config()
      return base_config

    def call(self, x):
      ret = tf.concat([x, tf.norm(x, axis=-1, keepdims=True)], axis=-1)
      diff = tf.subtract(ret[:, 1:, :, :], ret[:, :-1, :, :] )
      ##diff_diff = diff[:, 1:, :, :] - diff[:, :-1, :, :]
      
      ##before was bad feature
      diff_norm = tf.norm(diff[:, :, :, :2], axis=-1, keepdims=True)
      ##diff_norm = tf.norm(diff, axis=-1, keepdims=True)
      diff_norm = tf.where( tf.math.equal(diff_norm, 0), 1.0, diff_norm)
      cos_x = tf.divide( x[:, 1:, :, :1], diff_norm )
      cos_x =tf.pad(cos_x,  [[0, 0], [1, 0], [0, 0], [0, 0] ], 'CONSTANT')
      cos_y = tf.divide( x[:, 1:, :, 1:2],  diff_norm )
      cos_y =tf.pad(cos_y,  [[0, 0], [1, 0], [0, 0], [0, 0] ], 'CONSTANT')

      diff =tf.pad(diff,  [[0, 0], [1, 0], [0, 0], [0, 0] ], 'CONSTANT')

      tan = tf.divide( x[:, :, :, 1:2], 
              tf.where( tf.math.equal(x[:, :, :, :1], 0), 1.0,  x[:, :, :, :1]) )

      ret = tf.concat([ret, diff, cos_x, cos_y, tan], axis=-1)
      ret = self.bn(ret) 

      return ret

class LandmarkEmbedding(Layer):
    def __init__(self, length, emb_length, shape, filters=2, **kwargs) -> None:
      super(LandmarkEmbedding, self).__init__(**kwargs)
      self.dense = tf.keras.layers.Dense(length, activation='linear', kernel_initializer=INIT_GLOROT_UNIFORM)
      self.gelu = tf.keras.activations.gelu
      self.out_emb = tf.keras.layers.Dense(emb_length, use_bias=False, kernel_initializer=INIT_HE_UNIFORM)
      self.reshape = tf.keras.layers.Reshape((-1, shape))

      #self.conv = tf.keras.layers.Conv1D(filters, 3, padding='same')

      self.length = length
      self.emb_length = emb_length
      self.shape = shape
      self.filters = filters

    def get_config(self):
      base_config = super(LandmarkEmbedding, self).get_config()
      base_config['length'] = self.length
      base_config['emb_length'] = self.emb_length
      base_config['shape'] = self.shape
      base_config['filters'] = self.filters
      return base_config

    def call(self, x):
      #x = self.gelu( self.conv( self.reshape(x) ))
      #x = self.gelu( self.dense(x) )
      x = self.gelu(  self.dense( self.reshape(x) )  ) 
      return self.out_emb(x)

# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
 
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
 
        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask
 
        # Computing the weights by a softmax operation
        weights = softmax(scores)
 
        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)
 
# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, d_model, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.attention = DotProductAttention()  # Scaled dot product attention
        self.heads = num_heads  # Number of attention heads to use
        self.d_k = key_dim  # Dimensionality of the linearly projected queries and keys
        self.d_v = key_dim  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        self.W_q = Dense(key_dim)  # Learned projection matrix for the queries
        self.W_k = Dense(key_dim)  # Learned projection matrix for the keys
        self.W_v = Dense(key_dim)  # Learned projection matrix for the values
        self.W_o = Dense(d_model)  # Learned projection matrix for the multi-head output
 
    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing: (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations: (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x
 
    def call(self, query, key, value, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(query), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(key), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(value), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Compute the multi-head attention output using the reshaped queries, keys and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)
 
        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)
 
        # Apply one final linear projection to the output to generate the multi-head attention
        # Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = MultiHeadAttention(**kwargs) #tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x, mask=None):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x, mask=mask)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=tf.keras.activations.gelu, kernel_initializer=INIT_GLOROT_UNIFORM),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(d_model, kernel_initializer=INIT_HE_UNIFORM)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context)

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
  
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               key_dim,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        d_model=d_model)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        d_model=d_model)

    self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, key_dim, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        d_model=d_model)

    self.ffn = FeedForward(d_model, dff, dropout_rate=dropout_rate)

  def call(self, x, mask=None):
    x = self.self_attention(x, mask=mask)
    x = self.ffn(x)
    return x

class NonZeroAvgPool(Layer):

  def __init__(self, axis, mask) -> None:
    super().__init__()
    self.axis = axis
    self.mask = mask
  
  def call(self, x, input, training=False):
    if training:
      rets = []
      shape = tf.shape(x)
      for i in range(shape[0]):
        mask = tf.cast( (input[i] != self.mask), tf.bool) 
        xx = tf.boolean_mask( x[i], mask )
        xx = tf.reshape(xx, (-1, shape[-1]) )
        rets.append(tf.reduce_mean(xx, axis=self.axis)[None, :])

      rets = tf.concat(rets, axis=0)
      return rets
    return tf.reduce_mean(x, axis=1)


class SAP(Layer):
   
  def __init__(self, dim, axis) -> None:
    super().__init__()
    self.h_dense = tf.keras.layers.Dense(dim, activation='tanh', name='attn_dense')
    self.w_dense = tf.keras.layers.Dense(1, use_bias=False, name='attn_weights')
    self.axis = axis
    self.dim = dim
    
  def call(self, x, mask=None):
    h = self.h_dense(x)
    w = self.w_dense(h)
    w = softmax(w+mask[:, :, None], axis=1)
    return tf.reduce_sum(x*w, axis=self.axis)

class MSAP(Layer):
   
  def __init__(self, dim, heads, axis) -> None:
    super().__init__()
    self.h_dense = tf.keras.layers.Dense(dim, activation='tanh', name='attn_dense')
    self.heads_w = []
    for i in range(heads):
      self.heads_w.append( tf.keras.layers.Dense(1, use_bias=False, name='attn_weights_' + str(i) ) )
    self.axis = axis
    self.heads = heads
    self.dim = dim
    
  def call(self, x, mask=None):
    h = self.h_dense(x)
    out = []
    for i in range(self.heads):
      w = self.heads_w[i](h)
      w = softmax(w+mask[:, :, None], axis=1)
      out.append( tf.reduce_sum(x*w, axis=self.axis) )
    return tf.concat(out, axis=-1)

class ASP(Layer):
   
  def __init__(self, dim, axis) -> None:
    super().__init__()
    self.h_dense = tf.keras.layers.Dense(dim, activation='tanh', name='attn_dense')
    self.w_dense = tf.keras.layers.Dense(1, use_bias=False, name='attn_weights')
    self.axis = axis
    self.dim = dim
    
  def call(self, x, mask=None):
    h = self.h_dense(x)
    w = self.w_dense(h)
    w = softmax(w+mask[:, :, None], axis=1)
    m = tf.reduce_sum(x*w, axis=self.axis)
    si = tf.math.sqrt( ( (tf.reduce_sum( (x**2) * w, axis=self.axis) ) - m**2 ) + 1e-5)
    si = tf.clip_by_value(si, clip_value_min=1e-4, clip_value_max=1e4)
    return tf.concat([m, si], axis=-1)


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, key_dim, num_heads,
               dff, length, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    #self.pos_embedding = PositionalEmbedding(
    #    length=length, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model, key_dim=key_dim,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x, mask=None):
    # `x` is token-IDs shape: (batch, seq_len)
    #x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    #x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, mask=mask)

    return x  # Shape `(batch_size, seq_len, d_model)`.

def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, length, d_model):
    super().__init__()
    self.d_model = d_model
    #self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=length, depth=d_model)

  #def compute_mask(self, *args, **kwargs):
  #  return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x, mask=None):
    length = tf.shape(x)[1]
    #x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    if length > self.pos_encoding.shape[0]:
      x = x[:, :self.pos_encoding.shape[0]] + self.pos_encoding[tf.newaxis, :length, :self.d_model]
    else:
      x = x + self.pos_encoding[tf.newaxis, :length, :self.d_model]
    return x

class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, key_dim, num_heads, dff, length,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(length=length,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, key_dim=key_dim, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]


  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    #x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)


    # The shape of x is (batch_size, target_seq_len, d_model).
    return x


if __name__ == '__main__':
  encoder = Encoder(num_heads=4, num_layers=2, d_model=64, key_dim=64, length=400, dff=256)
  x = np.random.normal(size=(1, 22, 256))
  inp = np.zeros((2, 600, 256))
  inp[0, :x.shape[1], :] = x
  inp[1, 2:2+x.shape[1], :] = x
  pool = NonZeroAvgPool(axis=0, mask=0)

  den = tf.keras.layers.Dense(64, use_bias=False)
  
  den_x = den(x)
  den_out = den(inp)
  enc_x =  encoder( den_x) 
  enc_out = encoder( den_out) 
  print(enc_x - enc_out[0, :22])

  out1 = pool(  encoder( den_x  ),  den_x, training=False )
  out =  pool(  encoder( den_out ), den_out, training=True )

  print(out1 - out[0]) 
  