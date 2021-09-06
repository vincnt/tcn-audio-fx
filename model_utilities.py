import torch
import torch.nn as nn

class GatedActivation(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
    def forward(self, input):
        out_hidden_split = torch.split(input, self.num_channels, dim=1)
        out = torch.tanh(out_hidden_split[0]) * torch.sigmoid(out_hidden_split[1])
        return out

class FlowThroughActivation(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input

class CondGatedActivation(nn.Module):
    def __init__(self, num_channels, num_cond):
        super().__init__()
        self.num_channels = num_channels//2
        self.conv = nn.Conv1d(in_channels=num_cond, out_channels=num_channels*2, kernel_size=1, padding=0, groups=1, bias=False)

    def forward(self, input, cond):
        out_hidden_split = torch.split(input, self.num_channels, dim=1)

        cond = cond.permute(0,2,1)
        cond = cond.expand(-1, -1, input.shape[-1])
        cond_out = self.conv(cond)

        cond_out_split = torch.split(cond_out, self.num_channels,  dim=1)

        out = torch.tanh(out_hidden_split[0] + cond_out_split[0]) * torch.sigmoid(out_hidden_split[1] + cond_out_split[1])
        return out

class DeepCondGatedActivation(nn.Module):
    def __init__(self, num_channels, num_cond):
        super().__init__()
        self.num_channels = num_channels
        self.conv = nn.Sequential(
          nn.Conv1d(in_channels=num_cond, out_channels=num_channels*2, kernel_size=1, padding=0, groups=1, bias=False),
          torch.nn.ReLU(),
          nn.Conv1d(in_channels=num_channels*2, out_channels=num_channels, kernel_size=1, padding=0, groups=1, bias=False),
          torch.nn.ReLU(),
          nn.Conv1d(in_channels=num_channels, out_channels=num_channels*2, kernel_size=1, padding=0, groups=1, bias=False),
        )

    def forward(self, input, cond):
        out_hidden_split = torch.split(input, self.num_channels, dim=1)

        cond = cond.permute(0,2,1)
        cond = cond.expand(-1, -1, input.shape[-1])
        cond_out = self.conv(cond)
        cond_out_split = torch.split(cond_out, self.num_channels,  dim=1)

        out = torch.tanh(out_hidden_split[0] + cond_out_split[0]) * torch.sigmoid(out_hidden_split[1] + cond_out_split[1])
        return out

class ApplyFiLM(torch.nn.Module):
  def __init__(self):
    super(ApplyFiLM, self).__init__()

  def forward(self, x, cond):
    g, b = torch.chunk(cond, 2, dim=-1) # split into g and b for linear function
    g = g.permute(0,2,1) # rearranges the dimensions
    b = b.permute(0,2,1)
    x = (x * g) + b
    return x

class BasicFiLM(torch.nn.Module):
  def __init__(self, embed_dim, num_cond_params):
    super(BasicFiLM, self).__init__()
    self.num_cond_params = num_cond_params
    self.embed_dim = embed_dim # number of latent features to project to
    self.net = torch.nn.Linear(num_cond_params, embed_dim * 2)

  def forward(self, x, cond):
    assert cond.shape[-1] == self.num_cond_params # for weird cuda broadcasting error
    cond = self.net(cond)
    g, b = torch.chunk(cond, 2, dim=-1) # split into g and b for linear function
    g = g.permute(0,2,1) # rearranges the dimensions
    b = b.permute(0,2,1)
    x = (x * g) + b
    return x

class ShallowFilm(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(ShallowFilm, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim, embed_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim, embed_dim*2),
    )

class DeepFiLM(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(DeepFiLM, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim * 4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*2),
    )

class DeeperFiLM(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(DeeperFiLM, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim * 10),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*10, embed_dim*5),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*5, embed_dim*10),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*10, embed_dim*2),
    )

class SiLUFiLM(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(SiLUFiLM, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim * 4),
      torch.nn.SiLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*4),
      torch.nn.SiLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*2),
    )

class BottleneckFiLM(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(BottleneckFiLM, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim * 4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*2),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*2, embed_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim, embed_dim*2),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*2, embed_dim*4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*2),
    )

class DeepBottleneckFiLM(BasicFiLM):
  def __init__(self, embed_dim, num_cond_params):
    super(DeepBottleneckFiLM, self).__init__(embed_dim, num_cond_params)
    self.net = torch.nn.Sequential(
      torch.nn.Linear(num_cond_params, embed_dim * 4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*8),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*8, embed_dim*4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*2),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*2, embed_dim),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim, embed_dim*2),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*2, embed_dim*4),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*4, embed_dim*8),
      torch.nn.ReLU(),
      torch.nn.Linear(embed_dim*8, embed_dim*2),
    )
            
class DeepSequential(nn.Module):
    def __init__(self, input, mid, output):
        super().__init__()
        self.net = torch.nn.Sequential(
              torch.nn.Linear(input, mid * 4),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*4, mid*4),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*4, output*2),
            )
    def forward(self, input):
        return self.net(input)

class DeeperSequential(nn.Module):
    def __init__(self, input, mid, output):
        super().__init__()
        self.net = torch.nn.Sequential(
              torch.nn.Linear(input, mid * 8),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*8, mid*4),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*4, mid),              
              torch.nn.ReLU(),
              torch.nn.Linear(mid, output*2),
            )
    def forward(self, input):
        return self.net(input)

class SimpleSequential(nn.Module):
    def __init__(self, input, mid, output):
        super().__init__()
        self.net = torch.nn.Sequential(
              torch.nn.Linear(input, mid * 2),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*2, mid*2),
              torch.nn.ReLU(),
              torch.nn.Linear(mid*2, output*2),
            )
    def forward(self, input):
        return self.net(input)