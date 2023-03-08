import importlib
import tensorflow as tf
import os
import shutil
from tensorflow.keras import optimizers
from dataset.dataloader import DataLoader
from utils import utils
from utils.logger import Logger
import numpy as np
from tqdm import tqdm

TMP_PATH = 'tmp/'

class Pipeline():

    def __init__(self, cfg) -> None:
        self.build_model(cfg)
        self.train_loader = DataLoader(cfg.train_loader, cfg.augment)
        self.epochs = cfg.epochs
        self.loss = getattr(importlib.import_module("losses.{}".format(cfg.model.loss.module))
                             , cfg.model.loss.name)
        self.optimizer = getattr(optimizers, cfg.optimizer.name)(
            getattr(utils, cfg.optimizer.schedule.name)(**cfg.optimizer.schedule.params) )
        
        self.logger = Logger(cfg.logger)
        if os.path.exists(TMP_PATH):
            shutil.rmtree(TMP_PATH)
        os.makedirs(TMP_PATH, exist_ok=True)
    
    def build_model(self, cfg):
        encoder = getattr(importlib.import_module("models.{}".format(cfg.model.module))
                             , cfg.model.name)(**cfg.model.params)
        self.norm_input = encoder.remove_trend
        inp = tf.keras.layers.Input(encoder.get_shape())
        out = encoder(inp)
        self.model = tf.keras.models.Model(inp, out)
        self.model.summary()
    
    def apply_gradients(self, nloss, tape):
        grads = tape.gradient(nloss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    def get_std(x, num):
        a = x[:, :, num, :]
        mask = a!=0
        a[mask]
    
    def train(self, e):

        size = len(self.train_loader)
        #for i in range(10):
        #    d = self.train_loader[i]
        #    a = 3

        overall_loss = 0
        for i, batch in enumerate(self.train_loader):
            X, X_aug, y_true = batch
            mask = X != 0 
            X = self.norm_input(X)

            with tf.GradientTape() as tape:
                y_pred = self.model(X_aug, training=True)
                nloss = self.loss([y_true, X_aug], y_pred, mask)
            self.apply_gradients(nloss, tape)
            overall_loss += nloss.numpy()
            print("loss: {} iteration: {}/{} epoch: {}/{}".format(overall_loss/(i+1), i, 
                size, e, self.epochs), end='\r')
                
        return overall_loss / (i + 1)

    def evaluate(self):
        pass

    def run(self):

        for e in range(self.epochs):
            train_loss = self.train(e)

            model_path = TMP_PATH + 'model_{}.tf'.format(e)
            self.model.save(model_path)
            self.logger.log_epoch({'train_loss' : train_loss}, 
                                  {'name': 'model_{}'.format(e), 'path': model_path[:-2] + 'zip'})
        
        self.logger.stop_run()