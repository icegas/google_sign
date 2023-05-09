#!/bin/bash
python run_pipeline.py -m augment.recon.augmentations.mix_samples.p=0.25 logger.run_name=te_mix_25
python run_pipeline.py -m augment.recon.augmentations.mix_samples.p=0.75 logger.run_name=te_mix_75
python run_pipeline.py -m augment.recon.augmentations.rotate.p=0.5 logger.run_name=te_rotate_50
python run_pipeline.py -m augment.recon.augmentations.xyz_scaling.p=0.5 logger.run_name=te_scaling_50
python run_pipeline.py -m model.params.dropout_rate=0.05 logger.run_name=te_dr_005
python run_pipeline.py -m model.params.dropout_rate=0.4 logger.run_name=te_dr_04
python run_pipeline.py -m model.params.num_heads=2 logger.run_name=te_heads_2
python run_pipeline.py -m model.params.num_heads=6 logger.run_name=te_heads_6
