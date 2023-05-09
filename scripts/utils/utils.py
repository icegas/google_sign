import numpy as np
import tensorflow as tf
import math

LEFT_HAND_IDX = slice(468, 489)
RIGHT_HAND_IDX = slice(522, 543)

LEFT_POSE_IDXS = [502, 504, 506, 508, 510]
RIGHT_POSE_IDXS =[503, 505, 507, 509, 511]

POSE_IDX = [
  14, 12, 16, 20, 18, 22,
8, 6, 5, 4, 10, 0, 1, 2, 3, 7, 9,
11, 13, 21, 19, 15, 17
]
#POSE_IDX = slice(489, 522)
NUMBER_OF_MODEL_FEATURES = 115
NUMBER_OF_FEATURES = 543
NUMBER_OF_CLASSES = 250
MASK_VALUE = -1
LIP = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
        ]

#LIP = [
#            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
#            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
#            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
#            95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
#    
#    130 ,161, 159, 157, 133, 154, 145, 163, 398, 381, 173, 153,
#    384, 386, 388, 373, 380, 385, 152, 377, 400, 378, 379, 384,387, 374,
#    365, 397, 288, 361, 323, 454, 356, 389, 251, 284,
#    332, 297, 338, 10, 109, 67, 103,54, 21, 162,34,
#    227, 137, 177, 215, 138, 136, 149, 176, 148, 152, 377, 400, 
#    
#    370,  248,  168, 385, 158, 390, 219, 327, 414, 189, 168,
#    112, 217, 277, 429
#        ]

def set_memory_growth():
  gpus = tf.config.experimental.list_physical_devices("GPU")
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)

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
  def __init__(self, d_model, warmup_steps=4000): #4000
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    self.lr = 1e-3

  def __call__(self, step):
    step = step + 27
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    self.lr = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    return self.lr

class CosSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, iterations, warmup_steps=4000, num_training_steps=160, num_cycles=0.50, lr_max=1e-3): #4000
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    self.num_training_steps = num_training_steps 
    self.num_cycles = num_cycles
    self.lr_max = lr_max
    self.lr = lr_max
    self.iterations = iterations

  def __call__(self, current_step):
    current_step = current_step + 28
    current_step = int(current_step / self.iterations)

    if current_step < self.warmup_steps:
        #if WARMUP_METHOD == 'log':
        #    return self.lr_max * 0.10 ** (self.warmup_steps - current_step)
        #else:
        return self.lr_max * 2 ** -(self.warmup_steps - current_step)
    else:
        progress = float(current_step - self.warmup_steps) / float(max(1, self.num_training_steps - self.warmup_steps))

        self.lr = max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress))) * self.lr_max
        return self.lr
  