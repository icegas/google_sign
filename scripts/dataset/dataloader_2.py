import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import json
from copy import deepcopy
from utils.augmentation import Augmentator
from utils.utils import NUMBER_OF_FEATURES, Normalizer, MASK_VALUE

class DataLoader():

    def __init__(self, cfg, aug_cfg, split, test=False, epochs=200) -> None:

        self.path = Path(cfg['path'])
        self.num_frames=cfg['num_frames']
        self.cols = cfg['cols']
        self.batch_size = cfg['batch_size']
        self.augmentator = Augmentator(aug_cfg.recon)
        self.augmentator_X = Augmentator(aug_cfg.X)
        self.ret_user_id = cfg['ret_user_id']
        self.data_path = cfg['data_path']

        self.split = split
        if split=='train':
            self.X = np.load(self.data_path + 'X.npy')
            self.y = np.load(self.data_path + 'y.npy').astype('int64')
            #self.X = np.load(self.data_path + 'train_test_X.npy')
            #self.y = np.load(self.data_path + 'train_test_y.npy').astype('int64')
        elif split == 'val':
            self.X = np.load(self.data_path + 'val_X.npy')
            self.y = np.load(self.data_path + 'val_y.npy').astype('int64')
            self.augmentator.use_augmentation = False
            self.augmentator_X.use_augmentation = False
            #self.batch_size = 1
            #self.num_frames = 0

        self.seq_id = np.arange(self.y.shape[0])
        if test:
            for i in range(8):
                self.seq_id = np.append(self.seq_id, self.seq_id)
        #with open(cfg.label_map_path, 'r') as f:
        #    self.label_map = json.load(f)
        self.depth = 250
        self.weights = self.y.shape[0] / (self.depth * np.bincount(self.y))

        self.idx_cls = {}
        for i in range(250):
            self.idx_cls[i] = []

        for i in range(len(self.y)):
            self.idx_cls[self.y[i]].append(i)
        self.create_seq_id()
        self.epoch = 0
        self.epochs = epochs

    def generator(self):
        if self.split == 'val':
            for i in range(self.seq_id.shape[0]):
               yield self[i]
        else:
            for i in self.samples_id:
                yield self[i]

    def create_seq_id(self):
        self.samples_id = []
        for _ in range(100*4): #250 * 4 = 1000
        #for _ in range(1): #250 * 4 = 1000
            for i in range(250):
                self.samples_id.append( np.random.choice( self.idx_cls[i] ) )

    def __iter__(self):
        
        self.epoch += 1
        loader = tf.data.Dataset.from_generator(self.generator,
                output_types=(tf.float32, tf.float32, tf.float32, tf.float32), output_shapes=(None, None, None, None))
        if self.num_frames == 0:
           loader = self.generator()
        else:
            loader = loader.batch(self.batch_size)

        for batch in loader:
            yield batch

    def __len__(self):
        if self.split == 'val':
            return int(self.seq_id.shape[0] / self.batch_size)
        else:
            return int(len(self.samples_id) / self.batch_size)

    def __getitem__(self, id_num):

        X, y = deepcopy(self.X[id_num]), tf.one_hot( self.y[id_num], self.depth )
        #X, _ = self.augmentator_X.augment(X )
        #X_aug, X_mask = self.augmentator.augment(deepcopy(X))
        X_aug, X_mask, y = self.augmentator.augment(X, self, y)
        isnan = np.isnan(X_aug)
        X_aug[ isnan] =  0
        X_mask[isnan] =  0
        X[np.isnan(X)] = 0
        return X, X_aug, X_mask, y