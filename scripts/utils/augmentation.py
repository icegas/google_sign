import numpy as np

MASK_VALUE = -3

class Augmentator():

    def __init__(self, cfg) -> None:
        self.augs_params = cfg.augmentations
        self.use_augmentation = cfg.use_augmentation
        self.set_augmentations()

    def set_augmentations(self):
        self.augs = {
            'time_mask' : self.time_mask,
            'feature_mask' : self.feature_mask,
            'feature_time_mask' : self.feature_time_mask,
            'channel_mask': self.channel_mask,
            'normal_noise' : self.normal_noise,
            'scaling' : self.scaling
        }

    def time_mask(self, input, params):
        time_end_mask = np.random.randint(params['tmin'], params['tmax'])
        first_zero = np.where(input==0)[0]
        if first_zero.shape[0] > 0:
            size = first_zero[0] - time_end_mask
            if size < 5:
                return input
            time_start_mask = np.random.randint(0, size) 
        else:
            time_start_mask = np.random.randint(0, input.shape[0] - time_end_mask)
        input[time_start_mask:time_start_mask+time_end_mask] = MASK_VALUE
        return input
    
    def feature_mask(self, input, params):
        if np.random.uniform() < params['p']:
            mask_end = np.random.randint(params['fmin'], params['fmax'])
            mask_start = np.random.randint(0, input.shape[0] - mask_end)
            
            input[mask_start:mask_end, :, :] = MASK_VALUE
        return input

    def feature_time_mask(self, input, params):
        if np.random.uniform() < params['p']:
            time_end_mask = np.random.randint(params['tmin'], params['tmax'])
            time_start_mask = np.random.randint(0, input.shape[1] - time_end_mask)
            mask_end = np.random.randint(params['fmin'], params['fmax'])
            mask_start = np.random.randint(0, input.shape[0] - mask_end)
            
            input[mask_start:mask_end, time_start_mask:time_start_mask+time_end_mask, :] = MASK_VALUE
        return input
    
    def normal_noise(self, input, params):
        if np.random.uniform() < params['p']:
            input += np.random.normal(0, 0.01, size=input.shape)
        return input
    
    def scaling(self, input, params):
        if np.random.uniform() < params['p']:
            input *= np.random.normal(1, 0.01, size=input.shape)
        return input
    
    def channel_mask(self, input, params):
        if np.random.uniform() < params['p']:
            input[:, :, np.random.randint(input.shape[-1])] = MASK_VALUE
        return input

    def augment(self, input):

        if self.use_augmentation:

            for aug_name, params in self.augs_params.items():
                input = self.augs[aug_name](input, params)
        return input