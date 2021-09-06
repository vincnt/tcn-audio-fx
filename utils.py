from pytorch_lightning.callbacks import ModelCheckpoint

def compute_receptive_field(kernel_pattern, dilation_pattern):
    """ Compute the receptive field in samples."""
    rf = 1
    for kernel_size, dilation in zip(kernel_pattern, dilation_pattern):
      rf += (kernel_size-1) * dilation
    return rf

def to_np(x):
  return x.detach().cpu().squeeze().numpy()
  
class CheckpointSaverCallback(ModelCheckpoint):
  def on_keyboard_interrupt(self, trainer, pl_module):
    print('CheckpointSaverCallback - Keyboard Interrupt. Best model path, best model score', self.best_model_path, self.best_model_score)
    pl_module.logger.experiment.log_model(f'best_model', self.best_model_path)
    pl_module.logger.experiment.log_parameter("best_model_path", self.best_model_path)
    pl_module.logger.experiment.end()
  
  def on_train_start(self, trainer, pl_module):
    super(CheckpointSaverCallback, self).on_train_start(trainer, pl_module)
    trainable_parameters = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
    pl_module.logger.experiment.log_parameter("trainable_params", trainable_parameters)
    # save before training
    local_model_path = pl_module.logger.save_dir+f"/checkpoints/epoch0.ckpt"
    trainer.save_checkpoint(local_model_path)
    pl_module.logger.experiment.log_model(f'epoch0', local_model_path)

  def on_train_end(self, trainer, pl_module):
    print('CheckpointSaverCallback - Train End. Best model path, best model score', self.best_model_path, self.best_model_score)
    super(CheckpointSaverCallback, self).on_train_end(trainer, pl_module)
    pl_module.logger.experiment.log_model(f'best_model', self.best_model_path)
    pl_module.logger.experiment.log_parameter("best_model_path", self.best_model_path)
    pl_module.logger.experiment.end()

  def on_validation_end(self, trainer, pl_module):
    super(CheckpointSaverCallback, self).on_validation_end(trainer, pl_module)
    epoch = pl_module.current_epoch
    if epoch in [1,2,3,5,10,25,50,75,100,150,200,500,750,1000,1500,2000]:
      print(f'Epoch {epoch}: Saving checkpoint, logging histogram.')
      local_model_path = pl_module.logger.save_dir+f"/checkpoints/epoch{epoch}.ckpt"
      trainer.save_checkpoint(local_model_path)
      pl_module.logger.experiment.log_model(f'epoch{epoch}', local_model_path)
