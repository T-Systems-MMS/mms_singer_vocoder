from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE, denorm_am_mel, dynamic_range_compression
from models import Generator

if torch.cuda.is_available():
    global_device = torch.device('cuda')
else:
    global_device = torch.device('cpu')

def load_checkpoint(h, filepath, custom_device=None):
    device = custom_device if custom_device is not None else global_device
    assert os.path.isfile(filepath)
    generator = Generator(h).to(device)

    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    generator.load_state_dict(checkpoint_dict['generator'])
    generator.eval()
    generator.remove_weight_norm()
    print("Complete.")
    return generator


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(h, generator, input_dir, output_dir, custom_device=None, num_mels=80):
    device = custom_device if custom_device is not None else global_device
        
    filelist = os.listdir(input_dir)

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            if not (filname.endswith(".pkl") or filname.endswith(".npy")):
                continue
            x = np.load(os.path.join(input_dir, filname), allow_pickle=True)
            # shutil.copyfile(os.path.join(input_dir, filname),f"/opt/waterfalls/data/vocoder/097_diff_test/{filname[:-4]}_am_output.pkl")
            x = x["mel"]
            print(x.shape)
            x = denorm_am_mel(x)
            x = dynamic_range_compression(x)

            if len(x.shape) == 2:
                choir_mode = False
                x = np.expand_dims(x,0)
            else:
                choir_mode = True

            if x.shape[1] != num_mels:
                x = np.transpose(x, (0,2,1))
            
            x = torch.FloatTensor(x).to(device)
            y_g_hat = generator(x)
            audio = y_g_hat
            if choir_mode:
                audio = audio.sum(dim=0).squeeze() # Sum across batch dimension
            else:
                audio = audio.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int32')


            output_file = os.path.join(output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, h.sampling_rate, audio)


def load_config(config_file):

    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    return h

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_mels_dir', default='test_mel_files')
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    h = load_config(config_file)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = load_checkpoint(h, a.checkpoint_file, device)

    inference(h, model, a.input_mels_dir, a.output_dir, device, a.num_mels)


if __name__ == '__main__':
    main()

