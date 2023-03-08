import numpy as np
import tensorflow as tf

NUMBER_OF_FEATURES = 543
NUMBER_OF_CLASSES = 250

class Normalizer():

  def __init__(self, cfg) -> None:
    self.stat = cfg.stat
    self.set_function()
    self.sub_mean = cfg.sub_mean
    self.norm = cfg.normalization
    self.preprocess = cfg.preprocess

  def set_function(self):
    self.funcs = {
      'std' : self.std_normalization
    }

  #Is possible to remove rolling mean instead of whole, maybe EMA
  def remove_trend(self, df, cols):
    for t in df.type.unique():
      for f in df.frame.unique():
        df.loc[(df.type==t) & (df.frame==f), cols] -= df.loc[(df.type==t) & (df.frame==f), cols].mean() 
  
  def std_normalization(self, df, cols):
    for t in df.type.unique():
      for c in cols:
        df.loc[df.type==t, c] /= self.stat[t][c]['std']
  
  def normalize(self, df, cols):
    if self.sub_mean:
      self.remove_trend(df, cols)

    if self.preprocess:
      self.funcs[self.norm](df, cols)
    return df

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  