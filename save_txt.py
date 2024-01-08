import argparse
from pathlib import Path
import os
import torch
import pickle
import numpy as np

def save_file(fname, data, module=[], layer_no=[], save_txt=False, save_hex=False, phase=[]):
    # print(f"Type of data: {type(data)}")
    if type(data) is dict:
        for _key in data.keys():
            _fname = fname + f'_{_key}'
            save_file(_fname, data[_key])

    else:
        filename = os.path.join(fname + '.txt')

        if torch.is_tensor(data):
            try:
                data = data.detach()
            except:
                pass
            data = data.numpy()

        outfile = open(filename, mode='w')
        outfile.write(f'{data.shape}\n')

        if len(data.shape) == 0:
            outfile.write(f'{data}\n')
        elif len(data.shape) == 1:
            for x in data:
                outfile.write(f'{x}\n')
        else:
            w, x, y, z = data.shape
            # if w != 0:
            #     Out_Path += f'img{w+1}'
            for _i in range(w):
                for _j in range(x):
                    for _k in range(y):
                        for _l in range(z):
                            _value = data[_i, _j, _k, _l]
                            outfile.write(f'{_value}\n')

        outfile.close()
        print(f'\t\t--> Saved {filename}')

def load_pickle(Pickle_Path):
    with open(Pickle_Path, 'rb') as handle:
        data = pickle.load(handle)
    return data

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Text Data')
    parser.add_argument('--data', dest='data',
                        default="default_data0", type=str)
    parser.add_argument('--out_name', dest='out_name',
                        default="default_data0_txt", type=str)

    args = parser.parse_args()
    return args
         
         

if __name__ == "__main__":
    args = parse_args()
    # Read new results
    loaded_data = load_pickle(str(args.data))
    if isinstance(loaded_data, list): loaded_data = torch.tensor(np.array([x.numpy() for x in loaded_data]))
    loaded_data = loaded_data.float().contiguous().view(-1).detach().cpu().numpy()
    save_file(args.data,loaded_data)