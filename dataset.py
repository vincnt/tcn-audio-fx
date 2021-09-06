from torch.utils.data import Dataset
import os
import numpy as np
import librosa

def parse_filename(filename):
  filename = filename.rstrip('.wav')
  name, params = filename.split('_fx')
  params = 'fx' + params
  params = params.split('_')
  param_dict = {}
  param_dict['name'] = name.split('=')[1]
  for param in params:
    p, val = param.split('=')
    param_dict[p] = float(val) if 'p' in p else val
  return param_dict 

class AudioDataset(Dataset):
    def __init__(self, raw_dir, fx_dir, conditioning=False, num_conditioning=3, mono=True, sample_rate=44100, specific_fx_name=None, fx_list=[], sample_duration=10):
        self.sample_rate = sample_rate
        self.raw_dir = raw_dir
        self.fx_dir = fx_dir
        self.mono = mono
        self.duration = sample_duration

        self.raw_paths = []
        self.fx_paths = []
        self.sample_info = []
        self.conditioning_arrays = []

        for file in os.listdir(fx_dir):
          filename = os.fsdecode(file)
          file_path = os.path.join(fx_dir, filename)

          params = parse_filename(filename)

          # if more than one fx, add one hot
          if specific_fx_name is None: 
            if params['fx'] in fx_list:
              conditioning_array = np.zeros(num_conditioning + len(fx_list), dtype=np.float32)
              conditioning_array[num_conditioning + fx_list.index(params['fx'])] = 1.0
            elif params['fx'] == 'clean':
              conditioning_array = np.zeros(num_conditioning + len(fx_list), dtype=np.float32)
            else:
              print('error this shouldnt show up. check AudioDataset')
          else:
            conditioning_array = np.zeros(num_conditioning, dtype=np.float32)

          for i in range(num_conditioning):
            conditioning_array[i] = params[f'p{i+1}']          

          if (specific_fx_name is None and params['fx'] in fx_list) or (specific_fx_name == params['fx']):
            self.raw_paths.append(raw_dir+'/'+params['name'])
            self.fx_paths.append(file_path)
            self.sample_info.append(params)
            self.conditioning_arrays.append(conditioning_array)

    def __len__(self):
        return len(self.fx_paths)

    def __getitem__(self, idx):
        x, sr = librosa.load(self.raw_paths[idx], sr=self.sample_rate, mono=self.mono)
        y, sr = librosa.load(self.fx_paths[idx], sr=self.sample_rate, mono=self.mono)
        c = self.conditioning_arrays[idx]
        if self.mono:
          x = np.expand_dims(x, axis=0)
          y = np.expand_dims(y, axis=0)
          c = np.expand_dims(c, axis=0)

        return x, y, c, self.sample_info[idx]

# class ImpulseDataset(Dataset):
#     def __init__(self, params,  sample_rate=44100):
#         self.sample_rate = sample_rate

#         self.xs = []
#         self.ys = []
#         self.conditioning_arrays = []

#         for (fx_name, p1, p2 ,p3 ) in params:
#           fx = get_fx_chain(fx_name, p1, p2 ,p3)
#           impulse = np.zeros(sample_rate*5, dtype=np.float32)
#           impulse[...,0] = 1.0
#           impulse_pred = fx(impulse)
#           self.xs.append(impulse)
#           self.ys.append(impulse_pred)
#           self.conditioning_arrays.append(np.array([p1,p2,p3], dtype=np.float32))
           
#     def __len__(self):
#         return len(self.xs)

#     def __getitem__(self, idx):
#         x = self.xs[idx]
#         y = self.ys[idx]
#         c = self.conditioning_arrays[idx]
#         x = np.expand_dims(x, axis=0)
#         y = np.expand_dims(y, axis=0)
#         c = np.expand_dims(c, axis=0)

#         return x, y, c
