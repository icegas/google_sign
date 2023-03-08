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
  

        #face_data_x = []
        #left_data_x = []
        #pose_data_x = []
        #right_data_x = []

        #face_data_y = []
        #left_data_y = []
        #pose_data_y = []
        #right_data_y = []

        #for i, batch in tqdm( enumerate(self.train_loader), total=len(self.train_loader) ):
        #    X, _, _ = batch
        #    mask_face_x = X[:, :, :468, 0] != 0
        #    mask_left_x = X[:, :, 468:489, 0] != 0
        #    mask_pose_x = X[:, :, 489:522, 0] != 0
        #    mask_right_x = X[:, :, 522:, 0] != 0

        #    mask_face_y = X[:, :, :468, 1] != 0
        #    mask_left_y = X[:, :, 468:489, 1] != 0
        #    mask_pose_y = X[:, :, 489:522, 1] != 0
        #    mask_right_y = X[:, :, 522:, 1] != 0
        #    
        #    out = self.norm_input(X).numpy()
        #    face_data_x.append(out[:, :, :468, 0][mask_face_x])
        #    left_data_x.append(out[:, :, 468:489, 0][mask_left_x])
        #    pose_data_x.append(out[:, :, 489:522, 0][mask_pose_x])
        #    right_data_x.append(out[:, :, 522:, 0][mask_right_x])

        #    face_data_y.append(out[:, :, :468, 1][mask_face_y])
        #    left_data_y.append(out[:, :, 468:489, 1][mask_left_y])
        #    pose_data_y.append(out[:, :, 489:522, 1][mask_pose_y])
        #    right_data_y.append(out[:, :, 522:, 1][mask_right_y])
        
        #print("face std: {}".format(  np.concatenate(face_data_x ).std()) )
        #print("left std: {}".format(  np.concatenate(left_data_x ).std()) )
        #print("pose std: {}".format(  np.concatenate(pose_data_x ).std()) )
        #print("right std: {}".format( np.concatenate(right_data_x).std()) )

        #print("face std: {}".format(  np.concatenate(face_data_y ).std()) )
        #print("left std: {}".format(  np.concatenate(left_data_y ).std()) )
        #print("pose std: {}".format(  np.concatenate(pose_data_y ).std()) )
        #print("right std: {}".format( np.concatenate(right_data_y).std()) )