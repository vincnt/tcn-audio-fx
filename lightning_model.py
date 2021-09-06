
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import auraloss
import os

from dataset import AudioDataset
from models import TCN, DamskaggWaveNet, BasicDDSP
from extra_models import OctopusTCN, OutputcoderTCN, NormedTCN, DeepOutputcoderTCN, ParallelTCN, Parallel2TCN, NotanhTCN

class EffectsNet(pl.LightningModule):
    def __init__(self, hparams):
        super(EffectsNet, self).__init__()

        # hparam preprocessing for backwards compatibbility
        conditioning_type = hparams['conditioning_type'] if 'conditioning_type' in hparams else 'basic_film'
        force_local_residual = hparams['force_local_residual'] if 'force_local_residual' in hparams else False
        conditioning_structure = hparams['conditioning_structure'] if 'conditioning_structure' in hparams else 'shallow'
        self.without_preemphasis = hparams['without_preemphasis'] if 'without_preemphasis' in hparams else False
        self.preemphasis_type = hparams['preemphasis_type'] if 'preemphasis_type' in hparams else 'aw'

        
        self.loss_functions_scaling = torch.tensor([hparams[f'{loss_fn}_scaling'] if f'{loss_fn}_scaling' in hparams else 1 for loss_fn in hparams['loss_functions']], dtype=torch.float32)
        
        if hparams['specific_fx_name']:
          num_conditioning = hparams["num_conditioning"]
        else:
          num_conditioning = hparams["num_conditioning"] + len(hparams["fx_list"])


        # instantiate models
        if 'ddsp' not in hparams['architecture']:
            if hparams['architecture'] == 'damskagg':
                chosen_model = DamskaggWaveNet
            elif hparams['architecture'] == 'tcn':
                chosen_model = TCN
            elif hparams['architecture'] == 'octopus':
                chosen_model = OctopusTCN
            elif hparams['architecture'] == 'outputcoder':
                chosen_model = OutputcoderTCN
            elif hparams['architecture'] == 'deep_outputcoder':
                chosen_model = DeepOutputcoderTCN
            elif hparams['architecture'] == 'normed_tcn':
                chosen_model = NormedTCN
            elif hparams['architecture'] == 'parallel_tcn':
                chosen_model = ParallelTCN
            elif hparams['architecture'] == 'parallel2_tcn':
                chosen_model = Parallel2TCN
            elif hparams['architecture'] == 'no_tanh_tcn':
                chosen_model = NotanhTCN

            self.net = chosen_model(
                num_channels=hparams["num_channels"],
                dilation_depth=hparams["dilation_depth"],
                num_repeat=hparams["num_repeat"],
                kernel_size=hparams["kernel_size"],
                dilation_factor=hparams["dilation_factor"],
                conditioning = hparams["conditioning"], 
                num_conditioning = num_conditioning,
                activation=hparams["activation"],
                grouping = hparams['grouping'],
                bias = hparams['bias'],
                conditioning_type = conditioning_type,
                force_local_residual=force_local_residual,
                conditioning_structure = conditioning_structure
            )
        else:
            if hparams['architecture'] == 'ddsp_basic':
                print('Using BasicDDSP model')
                self.net = BasicDDSP(
                    num_channels=hparams["num_channels"],
                    dilation_depth=hparams["dilation_depth"],
                    num_repeat=hparams["num_repeat"],
                    kernel_size=hparams["kernel_size"],
                    dilation_factor=hparams["dilation_factor"],
                    conditioning = hparams["conditioning"], 
                    num_conditioning = num_conditioning,
                    activation=hparams["activation"],
                    grouping = hparams['grouping'],
                    bias = hparams['bias'],
                    conditioning_type = conditioning_type,
                    sample_rate = hparams['sample_rate'])

        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.loss_functions = {
            'mae':torch.nn.L1Loss(),
            'esr':auraloss.time.ESRLoss(),
            'dc':auraloss.time.DCLoss(),
            'snr':auraloss.time.SNRLoss(),
            'stft':auraloss.freq.MultiResolutionSTFTLoss()
        }
        self.preemphasis = auraloss.perceptual.FIRFilter(filter_type=self.preemphasis_type)	# default is hp
        self.preemphasis.to(self.device)

        self.num_workers = 4 if self.device==torch.device('cuda:0') else 4


    def prepare_data(self):
        self.train_ds = AudioDataset(self.hparams.train_x_path, self.hparams.train_y_path, specific_fx_name=self.hparams.specific_fx_name,
                                     conditioning=self.hparams.conditioning, num_conditioning=self.hparams.num_conditioning, 
                                     sample_rate=self.hparams.sample_rate, sample_duration=self.hparams.sample_duration, fx_list=self.hparams.fx_list)
        self.valid_ds = AudioDataset(self.hparams.val_x_path, self.hparams.val_y_path, specific_fx_name=self.hparams.specific_fx_name, 
                                    conditioning=self.hparams.conditioning, num_conditioning=self.hparams.num_conditioning, 
                                     sample_rate=self.hparams.sample_rate, sample_duration=self.hparams.sample_duration, fx_list=self.hparams.fx_list)
        self.impulse_ds = AudioDataset(self.hparams.impulse_x_path, self.hparams.impulse_y_path, specific_fx_name=self.hparams.specific_fx_name,
                                     conditioning=self.hparams.conditioning, num_conditioning=self.hparams.num_conditioning, 
                                     sample_rate=self.hparams.sample_rate, sample_duration=self.hparams.sample_duration, fx_list=self.hparams.fx_list)
        # print('train_ds - len - path: ', len(self.train_ds), self.hparams.train_x_path)
        # print('val_ds - len - path: ', len(self.valid_ds), self.hparams.val_x_path)
        # for file in os.listdir('/dataset'):
        #     print('/dataset files', file)
        # count = 0
        # for file in os.listdir('/dataset/val_x'):
        #     count +=1
        # print('val_x count', count)
        # count = 0
        # for file in os.listdir('/dataset/val_y'):
        #     count +=1
        # print('val_y count', count)


    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        loaders = [
            DataLoader(self.valid_ds, batch_size=self.hparams.batch_size, num_workers=self.num_workers), 
            DataLoader(self.impulse_ds, batch_size=self.hparams.batch_size, num_workers=self.num_workers)]
        return loaders

    def forward(self, x, cond_params):
        return self.net(x, cond_params)

    def training_step(self, batch, batch_idx):
        x, y, cond_params, data = batch
        y_pred, data = self.forward(x, cond_params)
        
        if not self.without_preemphasis:
            y_pred, y = self.preemphasis(y_pred, y)

        train_loss = torch.tensor(0, dtype=torch.float32, requires_grad=True)
        for i, loss_fn in enumerate(self.hparams.loss_functions):
            if loss_fn != 'cosine':
                train_loss = train_loss + self.loss_functions_scaling[i] * self.loss_functions[loss_fn](y_pred, y)
            elif loss_fn == 'cosine':
                if data and 'pre_out' in data:
                    cos_sim = torch.nn.CosineSimilarity(dim=-1)
                    pre_out = data['pre_out']
                    num_channels = pre_out.shape[1]
                    total = num_channels * (num_channels-1) / 2
                    for c in range(num_channels):
                        for cc in range(c+1, num_channels):
                            cos_loss = cos_sim(pre_out[:,c,:], pre_out[:,cc,:]) / total
                            train_loss = train_loss - cos_loss

        self.log('hp/train_loss', train_loss, on_epoch=True, on_step=False, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
      x, y,  cond_params, data = batch

      y_pred, data = self.forward(x, cond_params)
    
      y_pred, y = self.preemphasis(y_pred, y)
      mae_loss = self.loss_functions['mae'](y_pred, y) 
      esr_loss = self.loss_functions['esr'](y_pred, y)
      stft_loss = self.loss_functions['stft'](y_pred, y)
      self.log('esr_val', esr_loss, on_epoch=True, prog_bar=False, logger=True)
      self.log('mae_val', mae_loss, on_epoch=True, prog_bar=False, logger=True)
      self.log('stft_val', stft_loss, on_epoch=True, prog_bar=False, logger=True)
      self.log('combined_val', esr_loss+mae_loss+stft_loss, on_epoch=True, prog_bar=True, logger=True)
      return mae_loss