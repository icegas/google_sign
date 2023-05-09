#SSL PIPELINE
import importlib
import tensorflow as tf
import os
import shutil
from tensorflow.keras import optimizers
from dataset.dataloader import DataLoader
from utils import utils
from utils.logger import Logger
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import uuid
from losses.losses import cce_loss
from utils.utils import MASK_VALUE

class Pipeline():

    def __init__(self, cfg) -> None:
        self.build_model(cfg)
        self.eval = cfg.eval
        if cfg.test:
            self.eval = False

        self.train_loader = DataLoader(cfg.train_loader, cfg.augment, split='train', test=cfg.test)
        if self.eval:
            self.val_loader = DataLoader(cfg.train_loader, cfg.augment, split='val')
        self.epochs = cfg.epochs
        self.loss = getattr(importlib.import_module("losses.{}".format(cfg.model.loss.module))
                             , cfg.model.loss.name)
        self.alpha = cfg.model.loss.alpha
        self.gain = cfg.model.loss.gain
        self.optimizer = getattr(optimizers, cfg.optimizer.name)(
            getattr(utils, cfg.optimizer.schedule.name)(**cfg.optimizer.schedule.params) )
        #self.optimizer = getattr(optimizers, cfg.optimizer.name)(**cfg.optimizer.params)
            
        
        self.logger = Logger(cfg.logger)
        self.tmp_path = 'exps/' + str(uuid.uuid4())
        os.makedirs(self.tmp_path, exist_ok=True)
    
    def build_model(self, cfg):
        model_builder = getattr(importlib.import_module("models.{}".format(cfg.model.module))
                             , cfg.model.name)(**cfg.model.params)
        self.model = model_builder.build_model()
        self.norm_input = model_builder.norm_input
        self.model.summary()

    def apply_gradients_2(self, ncce, nmse, tape_cce, tape_mse):
        grads_cce = tape_cce.gradient(ncce, self.model.trainable_variables,  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads_mse = tape_mse.gradient(nmse, self.model.trainable_variables,  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads = []
        for g_cce, g_mse, in zip(grads_cce, grads_mse):
            alpha = (tf.reduce_mean(g_cce) / tf.reduce_mean(g_mse) ) 
            if alpha > 1e6:
                alpha = 0
            if alpha == 0:
                alpha = 1
            grads.append(self.alpha*g_mse + g_cce)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def apply_gradients(self, nloss, tape):
        grads = tape.gradient(nloss, self.model.trainable_variables,  unconnected_gradients=tf.UnconnectedGradients.ZERO)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def train(self, e):

        true_signs, pred_signs = np.array([]), np.array([])
        size = len(self.train_loader)
        acc_score = 0
        #for i in range(200):
        #    d = self.train_loader[i]
        #    a = 3

        mse_loss = 0; cce_loss = 0

        for i, batch in enumerate(self.train_loader):
            #if i < 520:
            #    continue
            X, X_aug, X_mask, y_true = batch
            mask = X_mask != 0
            mask = tf.concat([ tf.gather(mask, indices=utils.LIP, axis=2), 
                            mask[:, :, 468:, :] ], axis=2)
            X = self.norm_input(X)

            with tf.GradientTape() as tape:
                    y_pred = self.model(X_aug, training=True)
                    mse, cce = self.loss([X, y_true], y_pred, mask, alpha=self.alpha, gain=self.gain)
                    if mse is not None and cce is not None:
                        nloss = self.alpha * mse + cce

            if mse is None:
                mse = 0
                self.apply_gradients(cce, tape)
                cce_loss += cce.numpy()
            elif cce is None:
                cce = 0
                self.apply_gradients(mse, tape)
                mse_loss+= mse.numpy().mean()
            else:
                self.apply_gradients(nloss, tape)
                cce_loss += cce.numpy()
                mse_loss+= mse.numpy().mean()

            #overall_loss += nloss.numpy().mean()
            true_signs = np.append(true_signs, tf.argmax(y_true, -1).numpy().reshape(-1))
            pred_signs = np.append(pred_signs, tf.argmax(y_pred[1], -1).numpy().reshape(-1))
            print("mse: {:.4f} cce: {:.4f} iteration: {}/{} epoch: {}/{}".format(mse_loss/(i+1), cce_loss / (i + 1), i, 
                size, e, self.epochs), end='\r')
        print()
        if y_pred[1].shape.rank == 2:
            acc_score = accuracy_score(true_signs, pred_signs)
                
        return acc_score, mse_loss / (i+1), cce_loss / (i+1)

    def evaluate(self):
        print("Evaluation...")
        val_loss, mse_loss, cce_loss = 0, 0, 0
        #val_loss =0
        true_signs, pred_signs = np.array([]), np.array([])
        for i, batch in tqdm( enumerate(self.val_loader), total=len(self.val_loader)):

            X, X_aug, X_mask, y_true = batch
            #X, y_true = tf.convert_to_tensor(X[None, :].astype('float32')), y_true[None, :]
            mask = X != 0
            #mask = X != -1
            mask = tf.concat([ tf.gather(mask, indices=utils.LIP, axis=2), 
                             mask[:, :, 468:, :] ], axis=2)

            y_pred = self.model(X, training=False)
            X = self.norm_input(X)
            mask = mask[:, :, :, :X.shape[-1]]

            mse, cce = self.loss([X, y_true], y_pred, mask)
            if mse is None:
                cce_loss += cce.numpy()
            elif cce is None:
                mse_loss += mse.numpy().mean()
            else:
                cce_loss += cce.numpy()
                mse_loss += mse.numpy().mean()

            true_signs = np.append(true_signs, tf.argmax(y_true, -1).numpy().reshape(-1))
            pred_signs = np.append(pred_signs, tf.argmax(y_pred[1], -1).numpy().reshape(-1))

        acc_score = 0
        if y_pred[1].shape.rank == 2:
            acc_score = accuracy_score(true_signs, pred_signs)
        
        return acc_score, mse_loss / (i+1), cce_loss / (i+1)

    def run(self):

        train_accs, val_accs, epochs = [], [], []
        for e in range(self.epochs):
            epochs.append(e)
            train_acc, train_mse, train_cce = self.train(e)
            train_accs.append(train_acc)
            log_metrics =  { 'train_accuracy' : train_acc, 'epoch' : e,
                            'train_accuracy_best' : np.asarray(train_accs).max(),
                            'train_mse' : train_mse, 'train_cce' : train_cce}


            if self.eval:
                val_acc, val_mse, val_cce = self.evaluate()
                val_accs.append(val_acc)
                log_metrics['val_accuracy'] = val_acc
                log_metrics['val_accuracy_best'] = np.asarray(val_accs).max()
                log_metrics['best_val_epoch'] = epochs[np.asarray(val_accs).argmax()]
                log_metrics['val_mse'] = val_mse
                log_metrics['val_cce'] = val_cce

            model_path = self.tmp_path + '/model_{}.tf'.format(e)
            self.model.save(model_path)
            self.logger.log_epoch(log_metrics, model_path)

            print("train accuracy: {:.4f} train mse: {:.4f} train cce: {:.4f}".format(
                train_acc, train_mse, train_cce))
            if self.eval:
                print("val accuracy: {:.4f}, val_mse: {:.4f}, val_cce: {:.4f}".format(val_acc, val_mse, val_cce))
                #print("val loss: {:.4f}, val accuracy: {:.4f} val mse: {:.4f} val cce: {:.4f}".format(
                #    val_loss, val_acc, val_mse, val_cce))
        
        self.logger.end_run()