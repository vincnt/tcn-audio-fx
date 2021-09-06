import torch
import torch.nn as nn

from models import BaseTCN, TCN, CausalConv1d, _conv_stack
from model_utilities import GatedActivation

class OutputcoderTCN(TCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(OutputcoderTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)
      self.output_layer = torch.nn.Sequential(
        CausalConv1d(in_channels=num_channels,out_channels=64,kernel_size=1,bias=bias),
        GatedActivation(num_channels=32),
        CausalConv1d(in_channels=32,out_channels=32,kernel_size=1,bias=bias),
        GatedActivation(num_channels=16),
        CausalConv1d(in_channels=16,out_channels=16,kernel_size=1,bias=bias),
        GatedActivation(num_channels=8),
        CausalConv1d(in_channels=8,out_channels=1,kernel_size=1,bias=bias)
      )

class DeepOutputcoderTCN(TCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(DeepOutputcoderTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)
      self.output_layer = torch.nn.Sequential(
        CausalConv1d(in_channels=num_channels,out_channels=64,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=64,out_channels=128,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=128,out_channels=64,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=64,out_channels=32,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=32,out_channels=8,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=8,out_channels=4,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=4,out_channels=num_channels,kernel_size=1,bias=bias),
        torch.nn.ReLU(),
        CausalConv1d(in_channels=num_channels,out_channels=1,kernel_size=1,bias=bias)
      )


class NormedTCN(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(NormedTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think

        for hidden, residual in zip(self.hidden, self.residuals):
            skip_x = out
            out_hidden = hidden(skip_x)
            if self.conditioning:
              out = self.film(out, cond_params)
            out = self.activ(out_hidden)
            out = torch.nn.functional.normalize(out, dim=-1)
            res = residual(skip_x)
            res = torch.nn.functional.normalize(res, dim=-1)
            out = out + residual(skip_x)
            out = torch.nn.functional.normalize(out, dim=-1)

        output = self.output_layer(out)
        output = torch.nn.functional.normalize(output, dim=-1)
        return output, None

class ParallelTCN(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(ParallelTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)
      self.num_channels = num_channels
      self.conditioning = conditioning

      dilations = [dilation_factor ** d for d in range(dilation_depth)] * num_repeat
      internal_channels = int(num_channels * 2) if 'gated' in activation else int(num_channels)
      if grouping == 'all':
        groups = 1
      elif grouping == 'local_out':
        groups = internal_channels
      else:
        groups = num_channels
      groups = 1 if grouping == 'all' else num_channels
      self.hidden_parallel = _conv_stack(dilations, num_channels, internal_channels, kernel_size, groups=groups, bias=bias)
      self.output_layer = CausalConv1d(in_channels=num_channels*2,out_channels=1,kernel_size=1,bias=bias)

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think

        for hidden, residual in zip(self.hidden, self.residuals):
            skip_x = out
            out_hidden = hidden(skip_x)
            if self.conditioning:
              out_hidden = self.film(out_hidden, cond_params)
            out = self.activ(out_hidden)
            out = out + residual(skip_x)
        
        parallel_out = self.input_layer(x)
        for hidden in self.hidden_parallel:
          parallel_out = hidden(parallel_out)
          if self.conditioning:
              parallel_out = self.film(parallel_out, cond_params)
          parallel_out = self.activ(parallel_out)

        combined_out = torch.cat([out, parallel_out],dim=1)
        data = {'pre_out':combined_out}
        output = self.output_layer(combined_out)
        return torch.tanh(output), data

# input into each layer rather than cumulative residual
class Parallel2TCN(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False):
      super(Parallel2TCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual)

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think
        skip_x = out

        for hidden, residual in zip(self.hidden, self.residuals):
            out_hidden = hidden(skip_x)
            out = self.activ(out_hidden)
            if self.conditioning:
              out = self.film(out, cond_params)
            out = out + residual(skip_x)

        pre_out_data = out
        output = self.output_layer(out)
        data = {'pre_out':pre_out_data}
        return torch.tanh(output), data

class NotanhTCN(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False):
      super(NotanhTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual)

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think

        for hidden, residual in zip(self.hidden, self.residuals):
            skip_x = out
            out_hidden = hidden(skip_x)
            out = self.activ(out_hidden)
            if self.conditioning:
              out = self.film(out, cond_params)
            out = out + residual(skip_x)

        pre_out_data = out
        output = self.output_layer(out)
        data = {'pre_out':pre_out_data}
        output = torch.nn.functional.normalize(output, dim=-1)
        return output, data


class OctopusTCN(BaseTCN):
  def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film', force_local_residual=False):
    super(OctopusTCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual)
    mix_channels = num_channels * dilation_depth * num_repeat
    groups = 1 if grouping == 'all' else num_channels
    self.output_mixer = nn.Conv1d(
        in_channels= mix_channels,
        out_channels=mix_channels,
        kernel_size=1,
        groups=groups
    )
    self.output_layer = nn.Conv1d(
        in_channels= mix_channels,
        out_channels=1,
        kernel_size=1,
    )

  def forward(self, x, cond_params):
      out = x
      out = self.input_layer(out) # this wasnt there before I think
      skip_x = out

      layer_outs = []
      for hidden, residual in zip(self.hidden, self.residuals):
          out_hidden = hidden(skip_x)
          if self.conditioning:
            out_hidden = self.film(out_hidden, cond_params)
          out = self.activ(out_hidden)
          layer_out = out + residual(skip_x)
          layer_outs.append(layer_out)

      out = torch.cat(layer_outs, dim=1)
      out = self.output_mixer(out)
      out = self.output_layer(out)
      return out, None