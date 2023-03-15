import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import json
from copy import deepcopy
from utils.augmentation import Augmentator
from utils.utils import NUMBER_OF_FEATURES, Normalizer

class DataLoader():

    def __init__(self, cfg, aug_cfg) -> None:
        self.manifest_df = pd.read_csv(cfg['manifest_path']).rename(columns={'participant_id' : 'user_id'})
        self.manifest_df['user_id'] = self.manifest_df['user_id'].apply(str)
        self.manifest_df = self.manifest_df[self.manifest_df.user_id.isin(["34503", "26734"])]
        
        if cfg.split=='train':
            self.manifest_df = self.manifest_df[~self.manifest_df.user_id.isin(cfg.val_users)]
        elif cfg.split == 'val':
            self.manifest_df = self.manifest_df[self.manifest_df.user_id.isin(cfg.val_users)]

        self.path = Path(cfg['path'])
        self.seq_id = self.manifest_df.sequence_id.values
        self.num_frames=cfg['num_frames']
        self.cols = cfg['cols']
        self.batch_size = cfg['batch_size']
        self.augmentator = Augmentator(aug_cfg)
        self.normalizer = Normalizer(cfg.preprocessing)

        with open(cfg.label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.depth = len( list(self.label_map.keys()) )

    def generator(self):
        for i in range(self.seq_id.shape[0]):
            yield self[i]

    def get_random_frames(self, df):
        unq_frames = df.frame.unique()
        ret = np.zeros( (self.num_frames, NUMBER_OF_FEATURES, len(self.cols)) )
        size = unq_frames.shape[0]

        if size <= self.num_frames:
            ret[:size, :] = df[self.cols].values.reshape((size, ret.shape[1], len(self.cols)))
        else:
            idx = np.random.randint(0, size-self.num_frames)
            norm_df = self.normalizer.normalize(
                df.loc[(df.frame >= unq_frames[idx]) & (df.frame < unq_frames[idx+self.num_frames])], self.cols )
            ret = norm_df[self.cols].values.reshape( (ret.shape) )
        return ret
    
    def __iter__(self):
        if self.num_frames == 0:
            pass
        else:
            dataloader = tf.data.Dataset.from_generator(self.generator,
                output_types=(tf.float32, tf.float32, tf.int32), output_shapes=(None, None, None)).shuffle(32).batch(self.batch_size)

        for batch in dataloader:
            yield batch

    def __len__(self):
        return int(self.manifest_df.shape[0] / self.batch_size)

    def __getitem__(self, id_num):
        row = self.manifest_df[self.manifest_df.sequence_id==self.seq_id[id_num]].copy()
        df = pd.read_parquet(self.path / row.path.values[0] )

        if self.num_frames == 0:
            return df[self.cols].fillna(0).values.reshape((543*len(self.cols), -1)), row.sign
        else:
            X = self.get_random_frames(df)
            X_aug = self.augmentator.augment(deepcopy(X))
            X[np.isnan(X)] = 0
            X_aug[np.isnan(X_aug)] = 0
            return X, X_aug, tf.one_hot(self.label_map[row.sign.iloc[0]], self.depth)