import numpy as np
from pysndfx import AudioEffectsChain
from argparse import ArgumentParser
import os
import soundfile as sf
import librosa

from sys import exit
from dataset import parse_filename

def get_fx_chain(choice, p1=0.5, p2=0.5, p3=0.5):
  if choice == 'reverb':
    fx = AudioEffectsChain() \
        .reverb(reverberance=p1*100,
                hf_damping=0,
                room_scale=p2*100,
                stereo_depth=100,
                pre_delay=p3*100,
                wet_gain=0,
                wet_only=False)
  elif choice =='overdrive':
    fx = AudioEffectsChain() \
          .overdrive(gain=p1*100, colour=p2*100)
  elif choice== 'delay':
    fx = AudioEffectsChain() \
                .delay(
                  gain_in=1,
                  gain_out=1,
                  delays=[p1*1000+1, p2*3000+1],
                  decays=[p3, p3],
                  parallel=False) 
  elif choice == 'eq':
    fx = AudioEffectsChain().equalizer(frequency=p1*10000+1, q=p2+0.5, db=p3*10 - 5)
  elif choice == 'compand':
    fx = AudioEffectsChain().compand(attack=p1*2, decay=p1*2+0.01, soft_knee=p2*10, threshold=p3*-50-20, db_from=-20.0, db_to=-20.0)
  elif choice == 'phaser':
    fx = AudioEffectsChain().phaser(gain_in=1,gain_out=1,delay=p1*2+1,decay=min(p2,0.99),speed=max(0.1,p3*2),triangular=False)
  elif choice == 'tremolo':
    fx = AudioEffectsChain().tremolo(freq=int(np.exp(p1+0.5)**3 - np.exp(0.5)**3), depth=max(0.1,p2*100))
  elif choice == 'pitch':
    fx = AudioEffectsChain() \
                .pitch(shift=p1*1000, # unit is cents: 100th of a semitone.
                  use_tree=False,
                  segment=82,
                  search=14.68,
                  overlap=12) # can do pitch but ALOT of noise
  elif choice == 'oververb':
    fx = AudioEffectsChain() \
      .overdrive(gain=20, colour=100) \
      .reverb(reverberance=100,
                hf_damping=0,
                room_scale=100,
                stereo_depth=100,
                pre_delay=0,
                wet_gain=0,
                wet_only=False)
  elif choice == 'none' or choice == 'clean':
    fx = lambda x, sample_in: x
  else:
    print('Effect not found.')
    return None
  return fx

def clear_files(dir):
  for file in os.listdir(dir):
    filename = os.fsdecode(file)
    file_path = os.path.join(dir, filename)
    if not os.path.isdir(file_path):
      os.remove(file_path)

def pad_files(in_dir, out_dir, sample_rate, mono, desired_duration=5):
  for file in os.listdir(in_dir):
    filename = os.fsdecode(file)
    print(filename)
    if (len(filename) - 4) > 15:
        out_filename = filename[:15] + '.wav'
    else:
        out_filename = filename
    in_file_path = os.path.join(in_dir, filename)
    out_file_path = os.path.join(out_dir, out_filename)
    if not os.path.isdir(in_file_path):
      x, sr = librosa.load(in_file_path, sr=sample_rate, mono=mono, duration=desired_duration)
      new_x = np.zeros(sample_rate*desired_duration)
      new_x[:x.shape[-1]] = x
      sf.write(out_file_path, new_x, sample_rate)

def create_dataset(input_dir, output_dir, generation_array, sample_rate=44100, mono=True, desired_duration=10):
  for file in os.listdir(input_dir):
    filename = os.fsdecode(file)
    print(filename)
    file_path = os.path.join(input_dir, filename)
    if os.path.isdir(file_path):
      continue
    x, sr = librosa.load(file_path, sr=sample_rate, mono=mono, duration=desired_duration)
    for i, (fx_name, p1, p2, p3) in enumerate(generation_array):
      p1 = np.round(p1,3)
      p2 = np.round(p2,3)
      p3 = np.round(p3,3)
      fx = get_fx_chain(fx_name, p1, p2, p3)
      y = fx(x, sample_in=sr)[...,:x.shape[-1]] # limit y maximum length to x length
      monostereo = 'mono' if mono else 'stereo'
      name = f'name={filename}_fx={fx_name}_p1={p1}_p2={p2}_p3={p3}.wav'
      sf.write(output_dir+'/'+name, y.T, sample_rate)

if __name__ == '__main__':
    # params = [('overdrive', i, 1, 0) for i in [0,0.5,1]]
    # params = [('reverb',i,1,0) for i in np.arange(0,1.01,0.1)]
    # params = [('reverb',i,1,0) for i in np.concatenate((np.arange(0,0.7,0.2),np.arange(0.7,1.01,0.02))) ]
    # params = [('oververb',0,0,0)]
    params = []
    for i in np.arange(0,1.01,0.2):
      for j in np.arange(0,1.01,0.2):
        params.append(('overdrive', i, j, 0))
        params.append(('reverb', i, j, 0))
        params.append(('delay', i, j, 0.5))
        params.append(('pitch',i,j,0))

    dataset_name = 'cond-mixed-reverb-overdrive-delay-pitch-noclean'
    dataset_comment = 'overdrive-reverb-delay mixed every 0.1 for p1 AND p2. no clean. '
    sample_rate=44100
    desired_duration = 20

    print(dataset_name, dataset_comment)
    print('params', params)
    continue_generation = input('continue generation? y for yes: ')
    if continue_generation != 'y':
      exit()

    cwd = os.getcwd()
    current_dir = os.path.join(cwd, 'datasets', dataset_name)
    if not os.path.isdir(current_dir):
        os.makedirs (current_dir)

    train_x_dir = current_dir + '/train_x'
    train_y_dir = current_dir + '/train_y'
    val_x_dir = current_dir + '/val_x'
    val_y_dir = current_dir + '/val_y'
    impulse_x_dir = current_dir + '/impulse_x'
    impulse_y_dir = current_dir + '/impulse_y'

    if not os.path.isdir(train_x_dir):
        os.makedirs(train_x_dir)
        os.makedirs(train_y_dir)
        os.makedirs(val_x_dir)
        os.makedirs(val_y_dir)
        os.makedirs(impulse_x_dir)
        os.makedirs(impulse_y_dir)

    print('padding files')
    pad_files(cwd+'/datasets/Train Audio', train_x_dir, sample_rate, mono=True, desired_duration=desired_duration)
    pad_files(cwd+'/datasets/Validation Audio', val_x_dir, sample_rate, mono=True, desired_duration=desired_duration)

    # create data
    print('---creating training audio')
    create_dataset(train_x_dir, train_y_dir, params, sample_rate=sample_rate, mono=True, desired_duration=desired_duration)
    print('---creating val audio')
    create_dataset(val_x_dir, val_y_dir, params, sample_rate=sample_rate, mono=True, desired_duration=desired_duration)

    print('--creating impulse audio')
    for (fx_name, p1, p2 ,p3 ) in params:
      fx = get_fx_chain(fx_name, p1, p2 ,p3)
      impulse = np.zeros(sample_rate*desired_duration, dtype=np.float32)
      impulse[...,0] = 1.0
      impulse_pred = fx(impulse, sample_in=sample_rate)[...,:impulse.shape[-1]]
      name = f'name=impulse.wav_fx={fx_name}_p1={p1}_p2={p2}_p3={p3}.wav'
      sf.write(impulse_x_dir+'/'+'impulse.wav', impulse, sample_rate)
      sf.write(impulse_y_dir+'/'+name, impulse_pred, sample_rate)
  
    with open(current_dir+"/comments.txt", "w") as f:
      f.write(dataset_comment)