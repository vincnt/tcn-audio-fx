from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np
from argparse import ArgumentParser
import os 

from utils import *
from models import *
from extra_models import *
from dataset import *
from lightning_model import EffectsNet

pl.seed_everything(0)

if __name__ == '__main__':
    cwd = os.getcwd()
    # print('cwd', cwd)
    cwd_files = os.listdir(cwd)
    print('cwd_files', cwd_files)
    root_files = os.listdir('/')
    print('root_files', root_files)
    dataset_files = os.listdir('/dataset/cond-mixed-reverb-overdrive-delay-pitch-noclean')
    print('dataset_files', dataset_files)
    home_files = os.listdir('/home')
    print('home_files', home_files)
    count = 0
    for file in os.listdir('/dataset/cond-mixed-reverb-overdrive-delay-pitch-noclean/val_x'):
        count +=1
    # print('val x count', count)
    count = 0
    for file in os.listdir('/dataset/cond-mixed-reverb-overdrive-delay-pitch-noclean/val_y'):
        # print(file)
        count +=1
    print('val y count', count)

    parser = ArgumentParser()
    parser.add_argument('--architecture', type=str, required=True)
    parser.add_argument('--loss_functions', help='delimited list input', type=lambda s: [item for item in s.split(',')], required=True)
    parser.add_argument('--esr_scaling', type=float, default=1)
    parser.add_argument('--mae_scaling', type=float, default=1)
    parser.add_argument('--stft_scaling', type=float, default=1)
    parser.add_argument('--specific_fx_name', type=str, required=True)
    parser.add_argument('--comments', type=str, required=True)
    parser.add_argument('--num_channels', type=int, default=12)
    parser.add_argument('--dilation_depth', type=int, required=True)
    parser.add_argument('--dilation_factor', type=int, required=True)
    parser.add_argument('--kernel_size', type=int, required=True)
    parser.add_argument('--num_repeat', type=int, default=1)
    parser.add_argument('--conditioning', default=False, action='store_true')
    parser.add_argument('--bias', default=False, action='store_true')
    parser.add_argument('--num_conditioning', type=int, default=3)
    parser.add_argument('--fx_list', help='delimited list input', type=lambda s: [item for item in s.split(',')])
    parser.add_argument('--activation', type=str, default='prelu')
    parser.add_argument('--grouping', type=str, default='local')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=4e-3)
    parser.add_argument('--gpus', type=int, default=-1)
    parser.add_argument('--tpu_cores', type=int, default=None)
    parser.add_argument('--cpu', default=False, action='store_true')
    parser.add_argument('--sample_duration', type=int, default=10)
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.getcwd(), "data"))
    parser.add_argument('--conditioning_type', type=str, default='basic_film')
    parser.add_argument('--force_local_residual', default=False, action='store_true')
    parser.add_argument('--without_preemphasis', default=False, action='store_true')
    parser.add_argument('--preemphasis_type', type=str, default='aw')
    parser.add_argument('--conditioning_structure', type=str, default='shallow')

    args = parser.parse_args()

    hparams = vars(args)
    hparams['train_x_path'] = os.path.join(args.data_dir, 'train_x')
    hparams['train_y_path'] = os.path.join(args.data_dir, 'train_y')
    hparams['val_x_path'] = os.path.join(args.data_dir, 'val_x')
    hparams['val_y_path'] = os.path.join(args.data_dir, 'val_y')
    hparams['impulse_x_path'] = os.path.join(args.data_dir, 'impulse_x')
    hparams['impulse_y_path'] = os.path.join(args.data_dir, 'impulse_y')

    if len(args.specific_fx_name) == 0:
        hparams['specific_fx_name'] = None

    rf = compute_receptive_field(kernel_pattern=[hparams['kernel_size']]*hparams['dilation_depth'], 
                                dilation_pattern=[hparams['dilation_factor'] ** i for i in range(hparams['dilation_depth'])]*hparams['num_repeat'])
    rf_s =  np.round(rf/hparams['sample_rate'], 2)
    hparams['rf'] = rf
    hparams['rf_s'] = rf_s
    print(f"Receptive field: {rf} samples or {rf_s} s")

    if args.fx_list and len(args.fx_list) > 0:
        args.specific_fx_name = None

    if args.specific_fx_name:
        assert args.specific_fx_name in args.comments
    else:
        assert len(args.fx_list) > 0
    if 'conditioning' in args.comments or 'cond' in args.comments or 'mixed' in args.comments:
        assert args.conditioning == True
    else:
        assert args.conditioning == False

    model = EffectsNet(hparams)

    if args.specific_fx_name:
        name = f'{args.specific_fx_name}-{args.architecture}-{args.rf_s}rf-{str(args.conditioning)}Cond'
    else:
        temp = ','.join(args.fx_list)
        name = f'mixed-{temp}-{args.architecture}-{args.rf_s}rf'


    # early_stop_callback = EarlyStopping(monitor="combined_val/dataloader_idx_0", patience=40, verbose=False, check_finite=True)
    checkpoint_callback = CheckpointSaverCallback(monitor="combined_val/dataloader_idx_0", every_n_epochs=1, save_top_k=1, auto_insert_metric_name=True)

    trainer = pl.Trainer(
        gpus=None if args.cpu else args.gpus,
        tpu_cores=args.tpu_cores,
        log_every_n_steps=10,
        # callbacks=[checkpoint_callback, early_stop_callback],
        callbacks=[checkpoint_callback],
        precision=16,
        max_time="00:04:00:00", # 1 hours
        progress_bar_refresh_rate=25
    )

    trainer.fit(model)