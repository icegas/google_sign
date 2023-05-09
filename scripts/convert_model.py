import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import pandas as pd
import numpy as np
from models.transformer_encoder import Transformer
#from models.transformer_ED_clf import Transformer
#from models.transformer_recon_clf import Transformer
import tflite_runtime.interpreter as tflite
from models.transformer_utils import *
from time import time

ROWS_PER_FRAME = 543
NAN_VALUE = 0.0
LENGTH = 64

# TFLite model for submission
class TFLiteModel(tf.Module):
    def __init__(self, model_path):
        super(TFLiteModel, self).__init__()

        # Load the feature generation and main models
        self.preprocess_layer = PreprocessLayer(LENGTH, ROWS_PER_FRAME)
        self.model = tf.keras.models.load_model(model_path)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, ROWS_PER_FRAME, 3], dtype=tf.float32, name='inputs')])
    def __call__(self, inputs):
        # Preprocess Data
        nans = tf.math.is_nan(inputs)
        x = tf.where(nans, NAN_VALUE, inputs)
        x = self.preprocess_layer(x)
        x = self.model(x[None, :])
        # Squeeze Output 1x250 -> 250
        outputs = x[1][0]

        # Return a dictionary with the output tensor
        return {'outputs': outputs}

class PreprocessLayer(tf.keras.layers.Layer):
  def __init__(self, length, rows_per_frame):
    super(PreprocessLayer, self).__init__()
    self.length = length
    self.rows_per_frame = rows_per_frame

  def call(self, x):
    shape = tf.shape(x)[0]
    x = tf.pad(x,  [[0, tf.reduce_max([0, self.length-shape]) ], [0, 0], [0, 0] ], 'CONSTANT') 
    #x = tf.cond(shape > self.length,
    #    true_fn= lambda: tf.image.resize(x, (self.length, self.rows_per_frame)),
    #    false_fn=lambda: tf.pad(x,  [[0, self.length-shape ], [0, 0], [0, 0] ], 'CONSTANT') )

    return x

def load_relevant_data_subset(pq_path, fillna=False):
    data_columns = ['x', 'y', 'z'] #'z'
    data = pd.read_parquet(pq_path, columns=data_columns)
    if fillna:
        data = data.fillna(NAN_VALUE)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))

    if fillna:
        ret = np.zeros((LENGTH, 543, 3))
        ret[:data.shape[0], :] = data
        data = ret

    return data.astype(np.float32)

#!!!!! Add tf.image.resize
def set_new_model(model, model_path):
    inp = tf.keras.layers.Input((543, 3), name='inputs')
    nans = tf.math.is_nan(inp)
    x = tf.where(nans, NAN_VALUE, inp)
    x = PreprocessLayer(length=LENGTH, rows_per_frame=ROWS_PER_FRAME, model_path=model_path)(x)

    #inter_model = tf.keras.models.Model(model.input,
    #                    model.get_layer('outputs').output)
    #out = inter_model(x[None, :])
    #model_inp = tf.convert_to_tensor(x[None, :])
    #out = model(model_inp)['outputs']
    out = tf.keras.layers.Lambda(lambda x: x, name='outputs')(x[1])
    return tf.keras.models.Model(inp, out)


def convert():
    pre_path = '/home/icegas/Desktop/kaggle/google_sign/mlartifacts/'
    ml_path =   '478560750682151201/eea28b87057c482eb28f3283aa356362/artifacts/model_92.tf/model_92.tf'
    model_path = pre_path + ml_path
    sp = ml_path.split('/')
    savename = sp[1] + '_' + sp[3].split('_')[1].split('.')[0]
    
    frames_path = '../../data/train_landmark_files/2044/994104770.parquet'

    frames = load_relevant_data_subset(frames_path, fillna=True)
    model = tf.keras.models.load_model(model_path)

    #encoder = Transformer(num_layers=1, d_model=256, key_dim=256, num_heads=4, dff=256, length=128, target_length=2)
    #model = encoder.build_model()
    #model.summary()
    #model.save('tmp.tf')
    #model = tf.keras.models.load_model('tmp.tf')

    out_0 = model(frames[None, :])[1]
    print(out_0[0].numpy().argmax())

    frames = load_relevant_data_subset(frames_path, fillna=False)
    model = TFLiteModel(model_path)
    out_1 = model(frames)['outputs'].numpy()
    print(out_1.argmax())

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    print("ERROR: {}".format( np.abs( np.asarray(out_0 - out_1).mean() ) ) )

    model_path = '../../notebooks/models/{}.tflite'.format(savename)
    with open(model_path, 'wb') as f:
        f.write(tflite_model)
    
    interpreter = tflite.Interpreter(model_path)
    prediction_fn = interpreter.get_signature_runner("serving_default")
    t = time()
    output = prediction_fn(inputs=frames)
    print("Time")
    print(1000*(time() - t) )
    t = time()
    output = prediction_fn(inputs=frames)
    print("Time")
    print(1000*(time() - t) )

    out_2 = output['outputs']
    print("ERROR: {}".format( np.abs( np.asarray(out_1 - out_2) ).mean() ) )
    print(out_2.argmax())

    inp1 = np.zeros((900, 543, 3)).astype('float32')
    out1 = prediction_fn(inputs=inp1)
    inp1 = np.zeros((1, 543, 3)).astype('float32')
    out2 = prediction_fn(inputs=inp1)
    a = 3

if __name__=='__main__':
    convert()