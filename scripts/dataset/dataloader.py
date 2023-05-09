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

    def __init__(self, cfg, aug_cfg, split, test=False) -> None:
        self.manifest_df = pd.read_csv(cfg['manifest_path']).rename(columns={'participant_id' : 'user_id'})
        self.manifest_df['user_id'] = self.manifest_df['user_id'].apply(str)
        if test:
            self.manifest_df = self.manifest_df[self.manifest_df.user_id.isin(["26734"])].iloc[:60]
            split = 'test_mode'

        self.path = Path(cfg['path'])
        self.num_frames=cfg['num_frames']
        self.cols = cfg['cols']
        self.batch_size = cfg['batch_size']
        self.augmentator = Augmentator(aug_cfg.recon)
        self.augmentator_X = Augmentator(aug_cfg.X)
        self.ret_user_id = cfg['ret_user_id']

        self.split = split
        if split=='train':
            self.manifest_df = self.manifest_df[~self.manifest_df.user_id.isin(cfg.val_users)]
        elif split == 'val':
            self.manifest_df = self.manifest_df[self.manifest_df.user_id.isin(cfg.val_users)]
            self.augmentator.use_augmentation = False
            self.augmentator_X.use_augmentation = False
            #self.batch_size = 1
            #self.num_frames = 0

        self.seq_id = self.manifest_df.sequence_id.unique()
        if test:
            for i in range(8):
                self.seq_id = np.append(self.seq_id, self.seq_id)
        with open(cfg.label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.depth = len( list(self.label_map.keys()) )

        self.idx_cls = {}
        for i in range(len(self.y)):
            self.idx_cls[self.y[i]] = i

    #def get_data()

    def generator(self):
        
        #for _ in range(4):
        #    for i in range(250)
        if self.split == 'val':
            for i in range(self.seq_id.shape[0]):
               yield self[i]
        else:
            for i in self.samples_id:
                yield self[i]

    def get_random_frames(self, df):
        unq_frames = df.frame.unique()
        ret = np.zeros( (self.num_frames, NUMBER_OF_FEATURES, len(self.cols)) )
        size = unq_frames.shape[0]

        if size <= self.num_frames:
            ret[:size, :] = df[self.cols].values.reshape((size, ret.shape[1], len(self.cols)))
        else:
            ret[:self.num_frames, :] = df[self.cols].values.reshape((size, ret.shape[1], len(self.cols)))[:self.num_frames, :]
            #ret = tf.image.resize(df[self.cols].values.reshape(-1, NUMBER_OF_FEATURES, len(self.cols)), 
            #        (self.num_frames, NUMBER_OF_FEATURES)).numpy()
            #idx = np.random.randint(0, size-self.num_frames)
            #norm_df = df.loc[(df.frame >= unq_frames[idx])
            #     & (df.frame < unq_frames[idx+self.num_frames])]
            #ret = norm_df[self.cols].values.reshape( (ret.shape) )
        return ret
    
    def load_frame(self, df):
        x = df[self.cols].values.reshape((-1, NUMBER_OF_FEATURES, len(self.cols)))
        if x.shape[0] < self.num_frames:
            ret = np.zeros( (self.num_frames, NUMBER_OF_FEATURES, len(self.cols)) )
            ret[:x.shape[0], :] = x
            return ret
        return tf.image.resize(x, (self.num_frames, NUMBER_OF_FEATURES)).numpy()
    
    def __iter__(self):
        
        loader = tf.data.Dataset.from_generator(self.generator,
                output_types=(tf.float32, tf.float32, tf.float32, tf.int32), output_shapes=(None, None, None, None))
        if self.num_frames == 0:
           loader = self.generator()
        else:
            if self.split=='train':
               loader = loader#.shuffle(32)
            loader = loader.batch(self.batch_size)

        for batch in loader:
            yield batch
    
    def create_seq_id(self):
        self.samples_id = []
        for _ in range(80):
            for i in range(250):
                self.samples_id.append( np.random.choice( self.idx_cls[i] ) )
                


    def __len__(self):
        return int(self.seq_id.shape[0] / self.batch_size)

    def __getitem__(self, id_num):
        row = self.manifest_df[self.manifest_df.sequence_id==self.seq_id[id_num]].copy()
        df = pd.read_parquet(self.path / row.path.values[0] )

        if self.num_frames == 0:
            if self.ret_user_id:
                return df[self.cols].fillna(0).values.reshape((-1, NUMBER_OF_FEATURES, len(self.cols))), \
            tf.one_hot(self.label_map[row.sign.iloc[0]], self.depth), [row.user_id.values[0], self.seq_id[id_num]]
            else:
                return df[self.cols].fillna(0).values.reshape((-1, NUMBER_OF_FEATURES, len(self.cols))), \
            tf.one_hot(self.label_map[row.sign.iloc[0]], self.depth)
        else:
            #X = self.get_random_frames(df)
            X = self.load_frame(df)
            X, _ = self.augmentator_X.augment(X)
            X_aug, X_mask = self.augmentator.augment(deepcopy(X))
            isnan = np.isnan(X_aug)
            X_aug[ isnan] =  0#-1 #0
            X_mask[isnan] =  0#-1 #0
            X[np.isnan(X)] = 0#-1 #0
            #X_mask = X_mask == MASK_VALUE
            #X_mask = X_mask != 0
            #X_mask = X_mask != -1
            return X, X_aug, X_mask, tf.one_hot(self.label_map[row.sign.iloc[0]], self.depth)