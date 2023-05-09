import numpy as np
from utils.utils import MASK_VALUE, RIGHT_HAND_IDX, LEFT_HAND_IDX, POSE_IDX, LIP
from copy import deepcopy
import math
import tensorflow as tf
from scipy import interpolate

class Augmentator():

    def __init__(self, cfg) -> None:
        self.augs_params = cfg.augmentations
        self.use_augmentation = cfg.use_augmentation
        self.max_aug = cfg.max_aug
        self.set_augmentations()

    def set_augmentations(self):
        self.augs = {
            'time_mask' : self.time_mask,
            'hand_time_mask' : self.hand_time_mask,
            'time_identity' : self.time_identity,
            'normal_noise' : self.normal_noise,
            'hand_normal_noise' : self.hand_normal_noise,
            'scaling' : self.scaling,
            'hand_scaling' : self.hand_scaling,
            'channel_mask': self.channel_mask,
            'hand_channel_mask' : self.hand_channel_mask,
            'channel_shuffle' : self.channel_shuffle,
            'hand_channel_shuffle' : self.channel_hand_shuffle,
            'frame_shuffle' : self.frame_shuffle,

            'mirror_hands' : self.mirror_hands,
            'flip' : self.flip,
            'rotate' : self.rotate,
            'xyz_scaling' : self.xyz_scaling,

            'double_hand_noise' : self.double_hand_noise,
            'bad_hand_noise' : self.bad_hand_noise,

            'mix_samples' : self.mix_samples
        }
    
    #X
    def mirror_hands(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < params['p']:
            mask = np.isnan(input)
            mask_0 = input==0
            counter = 1

            inp_in = input[:, :, 0]
            inp_in[np.isnan(inp_in)] = 0
            m = np.mean( inp_in[inp_in != 0] )
            input[:, :, 0] = ( input[:, :, 0] - m) * -1
            tmp = deepcopy(input[:, LEFT_HAND_IDX, :])
            input[:, LEFT_HAND_IDX, :] = input[:, RIGHT_HAND_IDX, :]
            input[:, RIGHT_HAND_IDX, :] = tmp
            input[:, :, 0] += m

            tmp = deepcopy(mask[:, LEFT_HAND_IDX, :])
            mask[:, LEFT_HAND_IDX, :] = mask[:, RIGHT_HAND_IDX, :]
            mask[:, RIGHT_HAND_IDX, :] = tmp
            
            input[mask] = np.nan
            input[mask_0] = 0

        return input, mask_inp, counter
    
    def flip(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            first_zero = np.where(input==0)[0]
            if first_zero.shape[0] > 0:
                input[:first_zero[0]] = np.flip(input[:first_zero[0]], 0)
            else:
                input = np.flip(input, 0)

        return input, mask_inp
    
    #frotate(x, y) = ((x − 0.5) cos θ − (y − 0.5) sin θ + 0.5,
    #(y − 0.5) cos θ + (x − 0.5) sin θ + 0.5),
    def rotate(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < self.get_p( params['p'] ):
            theta = np.random.normal(0, params['theta_std']) #4
            rot_x = input[:, :, 0] * np.cos(np.deg2rad(theta) ) - (input[:, :, 1] ) * np.sin(np.deg2rad(theta)) 
            rot_y = input[:, :, 1] * np.cos(np.deg2rad(theta) ) - (input[:, :, 0] ) * np.sin(np.deg2rad(theta)) 
            input[:, :, 0] = rot_x
            input[:, :, 1] = rot_y
            counter = 1

        return input, mask_inp, counter
    
    def xyz_scaling(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < self.get_p( params['p'] ):

            input *= np.random.normal(1, params['scale'], size=(3)) #0.125
            counter = 1

        return input, mask_inp, counter
    
    #X_aug
    
    def get_mask(self, input, percent, ret=False):
        first_zero = np.where(input==0)[0]
        time_mask = np.zeros(shape=input.shape[0])
        ret_zero = input.shape[0] 

        if first_zero.shape[0] > 0:
            delta = max(0, int(first_zero[0] * percent ) - 1)
            if delta > 0:
                start = np.random.randint(0, first_zero[0] - delta)
                end = start + delta
                time_mask[start:end] = 1
                ret_zero = first_zero[0]

        else:
            delta = int(input.shape[0] * percent ) - 1
            start = np.random.randint(0, input.shape[0] - delta)
            end = start + delta
            time_mask[start:end] = 1
        if ret:
            return time_mask.astype(bool), ret_zero
        return time_mask.astype(bool)


    def time_mask(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < params['p']:
            
            counter = 1
            time_idx = self.get_mask(input, params['percent'])
            input[time_idx] = 0
            mask_inp[time_idx] = MASK_VALUE
        
        return input, mask_inp, counter

    def time_identity(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:

            time_idx = self.get_mask(input, params['percent'])
            mask_inp[time_idx] = MASK_VALUE
        
        return input, mask_inp
    
    def set_hand_mask(self, input, mask_inp, time_idx, func):
        input[time_idx,    LEFT_HAND_IDX] = func(input[time_idx, LEFT_HAND_IDX])
        mask_inp[time_idx, LEFT_HAND_IDX] = MASK_VALUE
        input[time_idx,    RIGHT_HAND_IDX] = func(input[time_idx, RIGHT_HAND_IDX])
        mask_inp[time_idx, RIGHT_HAND_IDX] = MASK_VALUE
        
        return input, mask_inp
    
    def hand_time_mask(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < params['p']:

            #time_idx = self.get_mask(input, params['percent'])
            time_idx = np.random.uniform(size=input.shape[0]) < params['percent']
            input, mask_inp = self.set_hand_mask(input, mask_inp, time_idx,
                                                 lambda x: 0)

        return input, mask_inp, counter

    def normal_noise(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < self.get_p( params['p'] ):
            time_idx = self.get_mask(input, params['percent'])
            input[time_idx] += np.random.normal(0, params['std'], size=input[time_idx].shape)
            mask_inp[time_idx] = MASK_VALUE
            counter = 1

        return input, mask_inp, counter

    def hand_normal_noise(self, input, mask_inp, params):

        counter = 0
        if np.random.uniform() < params['p']:
            counter = 1
            time_idx = self.get_mask(input, params['percent'])
            input, mask_inp = self.set_hand_mask(input, mask_inp, time_idx,
                            lambda x: x + np.random.normal(0, params['std'], size=x.shape))
        return input, mask_inp, counter

    #sometime non informative hand near informative hand doubled with some noise
    def double_hand_noise(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < params['p']:
            counter = 1
            if np.isnan(input[:, LEFT_HAND_IDX]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum():
                input[:,    RIGHT_HAND_IDX] = input[:, LEFT_HAND_IDX] * np.random.normal(1,
                                                 params['std'], size=input[:, LEFT_HAND_IDX].shape)
            else:
                input[:,    LEFT_HAND_IDX] = input[:, RIGHT_HAND_IDX] * np.random.normal(1,
                                                 params['std'], size=input[:, RIGHT_HAND_IDX].shape)
            mask_inp[:, RIGHT_HAND_IDX] = MASK_VALUE
            mask_inp[:, LEFT_HAND_IDX] = MASK_VALUE
        return input, mask_inp, counter
    
    #bad hand - hand without any information that should be empty
    def bad_hand_noise(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < params['p']:
            counter = 1
            if np.isnan(input[:, LEFT_HAND_IDX]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum():
                input[:,    RIGHT_HAND_IDX] = input[:, 507:508, :] + np.random.normal(0, 
                                        params['std'], size=input[:, RIGHT_HAND_IDX, :].shape)
            else:
                input[:,    LEFT_HAND_IDX] = input[:, 508:509, :] + np.random.normal(0, 
                                        params['std'], size=input[:, RIGHT_HAND_IDX, :].shape)    
            mask_inp[:, LEFT_HAND_IDX] = MASK_VALUE
            mask_inp[:, RIGHT_HAND_IDX] = MASK_VALUE
        return input, mask_inp, counter
    
    def scaling(self, input, mask_inp, params):
        counter = 0
        if np.random.uniform() < self.get_p( params['p'] ):
            counter = 1
            time_idx = self.get_mask(input, params['percent'])
            input[time_idx] *= np.random.normal(1, params['std'], size=input[time_idx].shape)
            mask_inp[time_idx] = MASK_VALUE
        return input, mask_inp, counter
    
    def hand_scaling(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx = self.get_mask(input, params['percent'])
            input, mask_inp = self.set_hand_mask(input, mask_inp, time_idx,
                            lambda x: x * np.random.normal(1, params['std'], size=x.shape))
        return input, mask_inp
    
    def channel_mask(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx = self.get_mask(input, params['percent'])
            input[time_idx, :, np.random.randint(input.shape[-1])] = MASK_VALUE
            mask_inp[time_idx] = MASK_VALUE
        return input, mask_inp

    def hand_channel_mask(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx = self.get_mask(input, params['percent'])
            if np.isnan(input[:, LEFT_HAND_IDX ]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum():
                input[time_idx,    LEFT_HAND_IDX, np.random.randint(input.shape[-1])] = MASK_VALUE
                mask_inp[time_idx, LEFT_HAND_IDX] = MASK_VALUE
            else:
                input[time_idx,    RIGHT_HAND_IDX, np.random.randint(input.shape[-1])] = MASK_VALUE
                mask_inp[time_idx, RIGHT_HAND_IDX] = MASK_VALUE
        
        return input, mask_inp

    def channel_shuffle(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx = self.get_mask(input, params['percent'])
            x = deepcopy(input[time_idx, :, 1])
            input[time_idx, :, 1] = input[time_idx, :, 0]
            input[time_idx, :, 0] = x
            mask_inp[time_idx] = MASK_VALUE
        return input, mask_inp

    def channel_hand_shuffle(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx = self.get_mask(input, params['percent'])
            input[time_idx, :, 1] = input[time_idx, :, 0]
            mask_inp[time_idx] = MASK_VALUE

            if np.isnan(input[:, LEFT_HAND_IDX ]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum():
                input[time_idx,    LEFT_HAND_IDX, 1] = input[time_idx, LEFT_HAND_IDX, 0]
                mask_inp[time_idx, LEFT_HAND_IDX] = MASK_VALUE
            else:
                input[time_idx,    RIGHT_HAND_IDX, 1] = input[time_idx, RIGHT_HAND_IDX, 0]
                mask_inp[time_idx, RIGHT_HAND_IDX] = MASK_VALUE
        return input, mask_inp
    
    def frame_shuffle(self, input, mask_inp, params):
        if np.random.uniform() < params['p']:
            time_idx, ret_zero = self.get_mask(input, params['percent'], ret=True)
            idx = np.where(time_idx)[0]
            new_idx = np.arange(ret_zero)
            new_idx = np.delete(new_idx, idx)
            x = deepcopy(input)

            for i in idx:
                input[i] = x[np.random.choice(new_idx)]

            mask_inp[time_idx] = MASK_VALUE
        return input, mask_inp
    

    def fill_values(self, hand):
        y = hand[:, 0, 0]
        if y[y!=0].shape[0] < 2:
            return
        x = np.linspace(1, hand.shape[0], (y[y!=0]).shape[0])
        x_new = np.linspace(1, hand.shape[0], hand.shape[0])
        for i in range(hand.shape[1]):
            y = hand[:, i, 0]
            f = interpolate.interp1d(x, y[y!=0])
            hand[y==0, i, 0] = f(x_new[y==0])

            y = hand[:, i, 1]
            f = interpolate.interp1d(x, y[y!=0])
            hand[y==0, i, 1] = f(x_new[y==0])
    
    def mix_samples(self, input, loader, y, params):
        if np.random.uniform() < self.get_p( params['p'] ):
            #idx = np.random.choice( loader.idx_cls[y.numpy().argmax()] )
            idx = np.random.choice(loader.samples_id)
            mix_sample = deepcopy(loader.X[idx])
            mix_y = tf.one_hot( deepcopy(loader.y[idx]), loader.depth )
            alpha = params['alpha']
            a1 = np.random.gamma(alpha, alpha)
            a2 = np.random.gamma(alpha, alpha)
            thr = a1 / (a1+a2)

            if ( ( (np.isnan(input[:, LEFT_HAND_IDX ]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum()) and
                (np.isnan(mix_sample[:, LEFT_HAND_IDX ]).sum() > np.isnan(mix_sample[:, RIGHT_HAND_IDX]).sum()) ) or
                ( (np.isnan(input[:, LEFT_HAND_IDX ]).sum() > np.isnan(input[:, RIGHT_HAND_IDX]).sum()) and
                (np.isnan(mix_sample[:, LEFT_HAND_IDX ]).sum() < np.isnan(mix_sample[:, RIGHT_HAND_IDX]).sum()) ) ): 
                mix_sample = self.mirror_hands(mix_sample, deepcopy(mix_sample), {'p' : 1.1})[0]
            
        
        
            new_shape = np.where(input)[0][-1] + 1
            mix_sample = tf.image.resize( mix_sample[:(np.where(mix_sample)[0][-1] + 1)], (new_shape, 543) ).numpy()
            
            #interpolate hands
            #rnd = np.random.randint(0, 3)

            #if rnd == 0 or rnd == 2:
            #    mix_sample[np.isnan(mix_sample)] = 0
            #    self.fill_values( mix_sample[:, RIGHT_HAND_IDX] )
            #    self.fill_values( mix_sample[:, LEFT_HAND_IDX] )
            #if rnd == 1 or rnd==2:
            #    input[np.isnan(input)] = 0
            #    self.fill_values( input[:, RIGHT_HAND_IDX] )
            #    self.fill_values( input[:, LEFT_HAND_IDX] )
            mix_sample = np.append(mix_sample, np.zeros((input.shape[0] - mix_sample.shape[0], 543, 3)), axis=0)


            input = thr*input + (1-thr)*mix_sample
            y = thr*y + (1-thr) * mix_y
        return input, y
    
    def mix_samples_2(self, input, loader, y, params):
        if np.random.uniform() < params['p']:
            #idx = np.random.choice( loader.idx_cls[y] )
            idx = np.random.choice(loader.samples_id)
            mix_sample = deepcopy(loader.X[idx])
            mix_y = tf.one_hot( deepcopy(loader.y[idx]), loader.depth )
            #alpha = params['alpha']
            #a1 = np.random.gamma(alpha, alpha)
            #a2 = np.random.gamma(alpha, alpha)
            #thr = a1 / (a1+a2)

            if ( ( (np.isnan(input[:, LEFT_HAND_IDX ]).sum() < np.isnan(input[:, RIGHT_HAND_IDX]).sum()) and
                (np.isnan(mix_sample[:, LEFT_HAND_IDX ]).sum() > np.isnan(mix_sample[:, RIGHT_HAND_IDX]).sum()) ) or
                ( (np.isnan(input[:, LEFT_HAND_IDX ]).sum() > np.isnan(input[:, RIGHT_HAND_IDX]).sum()) and
                (np.isnan(mix_sample[:, LEFT_HAND_IDX ]).sum() < np.isnan(mix_sample[:, RIGHT_HAND_IDX]).sum()) ) ): 
                mix_sample = self.mirror_hands(mix_sample, deepcopy(mix_sample), {'p' : 1.1})[0]
        
        
            new_shape = np.where(input)[0][-1] + 1
            #if new_shape < 10:
            #    return input, y
            mix_sample = tf.image.resize( mix_sample[:(np.where(mix_sample)[0][-1] + 1)], (new_shape, 543) ).numpy()
            mix_sample = np.append(mix_sample, np.zeros((input.shape[0] - mix_sample.shape[0], 543, 3)), axis=0)

            rnd = np.random.randint(0, 2)

            if rnd == 0:
                nans_r = np.isnan(mix_sample[:, RIGHT_HAND_IDX]).sum()
                nans_l = np.isnan(mix_sample[:, LEFT_HAND_IDX]).sum()
                not_nans_r = (~np.isnan(mix_sample[:, RIGHT_HAND_IDX])).sum()
                not_nans_l = (~np.isnan(mix_sample[:, LEFT_HAND_IDX])).sum()
                not_nans = max(not_nans_r, not_nans_l)
                nans = min(nans_r, nans_l)
                percent = nans / (not_nans + nans)
                if percent > 0.5:
                    thr = 0.2
                elif percent > 0.2:
                    thr = 0.3
                elif percent > 0.1:
                    thr = 0.4
                else:
                    thr = 1.0
                if thr > 0:
                    input[:, RIGHT_HAND_IDX] = mix_sample[:, RIGHT_HAND_IDX]
                    input[:, LEFT_HAND_IDX] = mix_sample[:, LEFT_HAND_IDX]
            else:
                input[:, POSE_IDX] = mix_sample[:, POSE_IDX]
                input[:, LIP] = mix_sample[:, LIP]
                thr = 0.7

            #input = thr*input + (1-thr)*mix_sample
            #y = thr*y + (1-thr) * mix_y
        return input, y
    
    def time_mix(self, input, params):
        if np.random.uniform() < params['p']:
            shape = np.where(input)[0][-1] + 1
            alpha = params['alpha']
            a1 = np.random.gamma(alpha, alpha, size=shape)
            a2 = np.random.gamma(alpha, alpha, size=shape)
            thrs = a1 / (a1+a2)

            for i in range(shape):
                mix_frame = i + np.random.randint(1, 3)
                if next_frame > shape:
                    next_frame = shape

                input[i] = thrs[i]*input[i] + (1-thrs[i]) * input[mix_frame]
        return input
            
    
    def get_p(self, p_max):

        return p_max
        #step = loader.epoch
        #num_cycles = 0.5
        #num_training_steps = self.epochs

        ##if self.step < 10:
        ##    return 0.0
        #if self.step > int(num_training_steps * 0.8):
        #    return 0.0

        #ret = []

    
        #progress = float(self.step) / float(max(1, num_training_steps))
        #return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * p_max 

    def augment(self, input, loader, y):

        #if you are adding noise or scaling with mask it will produce MASK_VALUE + noise
        #if it will give lower results, just change augmentaion in a way
        #to don't adding noise where mask already exists
        mask_inp = deepcopy(input)
        self.step = loader.epoch
        self.epochs = loader.epochs
        counter = 0
        if self.use_augmentation:

            input, y = self.mix_samples(input, loader, y, self.augs_params['mix_samples'])

            for aug_name, params in self.augs_params.items():
                if aug_name == 'mix_samples':
                    continue
                input, mask_inp, c = self.augs[aug_name](input, mask_inp, params)
                counter += c

                if counter > self.max_aug:
                    break
        return input, mask_inp, y