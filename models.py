import torch
import torch.nn as nn
import math

from model_utilities import * 

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, : -self.__padding]
        return result

def _conv_stack(dilations, in_channels, out_channels, kernel_size, groups, bias):
    return nn.ModuleList(
        [
            CausalConv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                dilation=d,
                kernel_size=kernel_size,
                groups=groups,
                bias = bias
            )
            for i, d in enumerate(dilations)
        ]
    )

class BaseTCN(torch.nn.Module):
    def __init__(self, num_channels, dilation_depth, num_repeat, 
              kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type='basic_film', force_local_residual=False, conditioning_structure='shallow'):
      super(BaseTCN, self).__init__()
      self.num_channels = num_channels
      self.conditioning = conditioning
      self.conditioning_structure = conditioning_structure

      dilations = [dilation_factor ** d for d in range(dilation_depth)] * num_repeat
      internal_channels = int(num_channels * 2) if 'gated' in activation else int(num_channels)
      if grouping == 'all':
        groups = 1
      elif grouping == 'local_out':
        groups = internal_channels
      else:
        groups = num_channels
      groups = 1 if grouping == 'all' else num_channels
      self.hidden = _conv_stack(dilations, num_channels, internal_channels, kernel_size, groups=groups, bias=bias)
      residual_groups = num_channels if force_local_residual else groups
      if force_local_residual:
        residual_groups = num_channels
      elif grouping == 'all':
        residual_groups = 1
      else:
        residual_groups = num_channels
      self.residuals = _conv_stack(dilations, num_channels, num_channels, kernel_size=1, groups=residual_groups, bias=bias)
      self.input_layer = CausalConv1d(
          in_channels=1,
          out_channels=num_channels,
          kernel_size=1,
          bias=bias
      )
      self.output_layer = CausalConv1d(
          in_channels=num_channels,
          out_channels=1,
          kernel_size=1,
          bias=bias
      )
      if conditioning:
        
        if conditioning_structure == 'shallow':
          if conditioning_type == 'deep_film':
            self.film = DeepSequential(input=num_conditioning, mid=num_channels, output=internal_channels)
          elif conditioning_type == 'deeper_film':
            self.film = DeeperSequential(input=num_conditioning, mid=num_channels, output=internal_channels)           
          else:
            self.film = SimpleSequential(input=num_conditioning, mid=num_channels, output=internal_channels)
          self.films = torch.nn.ModuleList([ApplyFiLM()] * dilation_depth)

        elif conditioning_structure == 'deep':
          if conditioning_type == 'basic_film':
            self.film = BasicFiLM
          elif conditioning_type == 'shallow_film':
            self.film = ShallowFilm
          elif conditioning_type == 'deep_film':
            self.film = DeepFiLM
          elif conditioning_type == 'deeper_film':
            self.film = DeeperFiLM
          elif conditioning_type == 'silu_film':
            self.film = SiLUFiLM
          elif conditioning_type == 'bottleneck_film':
            self.film = BottleneckFiLM
          elif conditioning_type == 'deep_bottleneck_film':
            self.film = DeepBottleneckFiLM
          if activation == 'cond_gated':
            self.film = CondGatedActivation
          elif activation == 'deep_cond_gated':
            self.film = DeepCondGatedActivation
          self.films = torch.nn.ModuleList([self.film(internal_channels, num_conditioning) for i in range(dilation_depth)] )
      else:
        self.films = [None] * dilation_depth

      if activation == 'gated':
        self.activ = GatedActivation(num_channels=num_channels)
      elif activation == 'prelu':
        self.activ= torch.nn.PReLU(num_parameters=num_channels)
      elif activation == 'tanh':
        self.activ = torch.nn.Tanh()
      elif activation == 'cond_gated' or activation == 'deep_cond_gated':
        self.activ = FlowThroughActivation()
      else:
        self.activ= torch.nn.ReLU()

    def forward(self, x, cond_params):
      pass
      return None, None

class TCN(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(TCN, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think

        if self.conditioning and self.conditioning_structure == 'shallow':
          cond_in = self.film(cond_params)
        else:
          cond_in = cond_params


        for hidden, residual, film in zip(self.hidden, self.residuals, self.films):
            skip_x = out
            out_hidden = hidden(skip_x)
            if self.conditioning:
              out_hidden = film(out_hidden, cond_in)
            out = self.activ(out_hidden)
            out = out + residual(skip_x)

        pre_out_data = out
        output = self.output_layer(out)
        data = {'pre_out':pre_out_data}
        return torch.tanh(output), data


class DamskaggWaveNet(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film',force_local_residual=False,conditioning_structure='shallow'):
      super(DamskaggWaveNet, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type,force_local_residual,conditioning_structure)
      self.output_layer = None
      self.linear_mix = nn.Conv1d(
          in_channels=num_channels * dilation_depth * num_repeat,
          out_channels=1,
          kernel_size=1,
      )

    def forward(self, x, cond_params):
        out = x
        skips = []
        out = self.input_layer(out)

        if self.conditioning and self.conditioning_structure == 'shallow':
          cond_in = self.film(cond_params)
        else:
          cond_in = cond_params

        for hidden, residual, film in zip(self.hidden, self.residuals, self.films):
            x = out
            out_hidden = hidden(x)
            if self.conditioning:
              out_hidden = film(out_hidden, cond_in)
            out = self.activ(out_hidden)

            skips.append(out)
            out = residual(out) + x

        # modified "postprocess" step:
        out = torch.cat(skips, dim=1)
        out = self.linear_mix(out)
        return out, None


class BasicDDSP(BaseTCN):
    def __init__(self, num_channels, dilation_depth, num_repeat, kernel_size=2, dilation_factor=2, conditioning=False, num_conditioning=3, activation='gated', grouping='all', bias=True, conditioning_type='basic_film', sample_rate=44100):
      super(BasicDDSP, self).__init__(num_channels, dilation_depth, num_repeat, kernel_size, dilation_factor, conditioning, num_conditioning, activation, grouping, bias, conditioning_type)
      self.sample_rate = sample_rate
      self.n_harmonic = 101
      self.n_bands = 65
      self.pre_harmonic = CausalConv1d(
          in_channels=num_channels,
          out_channels=2*self.n_harmonic,
          kernel_size=1,
          bias=bias
      )
      self.pre_subtractive = CausalConv1d(
          in_channels=num_channels,
          out_channels=self.n_bands,
          kernel_size=1,
          bias=bias
      )
      self.output_layer = None

    def harmonic_synth(self, pitch, amplitudes, sampling_rate):
      n_harmonic = amplitudes.shape[-1]
      omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
      omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
      signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
      return signal

    def remove_above_nyquist(self, amplitudes, pitch, sampling_rate):
      n_harmonic = amplitudes.shape[-1]
      pitches = pitch * torch.arange(1, n_harmonic + 1).to(pitch)
      aa = (pitches < sampling_rate / 2).float() + 1e-4
      return amplitudes * aa

    def amp_to_impulse_response(self, amp, target_size):
      amp = torch.stack([amp, torch.zeros_like(amp)], -1)
      amp = torch.view_as_complex(amp)
      amp = torch.fft.irfft(amp)

      filter_size = amp.shape[-1]

      amp = torch.roll(amp, filter_size // 2, -1)
      win = torch.hann_window(filter_size, dtype=amp.dtype, device=amp.device)

      amp = amp * win

      amp = nn.functional.pad(amp, (0, int(target_size) - int(filter_size)))
      amp = torch.roll(amp, -filter_size // 2, -1)

      return amp


    def fft_convolve(self, signal, kernel):
        signal = nn.functional.pad(signal, (0, signal.shape[-1]))
        kernel = nn.functional.pad(kernel, (kernel.shape[-1], 0))

        output = torch.fft.irfft(torch.fft.rfft(signal) * torch.fft.rfft(kernel))
        output = output[..., output.shape[-1] // 2:]

        return output

    def forward(self, x, cond_params):
        out = x
        out = self.input_layer(out) # this wasnt there before I think

        # first process tcn
        for hidden, residual in zip(self.hidden, self.residuals):
            skip_x = out
            out_hidden = hidden(skip_x)
            out = self.activ(out_hidden)
            if self.conditioning:
              out = self.film(out, cond_params)
            out = out + residual(skip_x)

        # use tcn output as input to harmonic synth
        harmonic_input = self.pre_harmonic(out).permute(0,2,1)
        pitch_input, amplitude_input = torch.split(harmonic_input, self.n_harmonic, dim=2)

        amplitude_input = self.remove_above_nyquist(
          amplitude_input,
          pitch_input,
          self.sample_rate,
        )

        harmonic_signal = self.harmonic_synth(pitch_input, amplitude_input, self.sample_rate)
        harmonic_signal = harmonic_signal.permute(0,2,1)

        noise_input = self.pre_subtractive(out).permute(0,2,1)
        impulse = self.amp_to_impulse_response(noise_input, 1)
        noise = torch.rand(
            impulse.shape[0],
            impulse.shape[1],
            1,
        ).to(impulse) * 2 - 1
        noise = self.fft_convolve(noise, impulse).contiguous()
        noise = noise.permute(0,2,1)
        
        output = harmonic_signal + noise
        return output