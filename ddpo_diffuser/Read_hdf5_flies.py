import h5py
# import torch
import numpy as np


def read_hdf5_files(task):
    ob_values = {}
    with h5py.File("./dataset/datasets/" + task + ".hdf5", "r") as f:
        for key in f.keys():
            value = f[key]
            try:
                for sub_key in value.keys():
                    ob_values[sub_key] = np.array(value[sub_key])
            except:
                ob_values[key] = np.array(value)

    return ob_values


if __name__ == '__main__':
    task = 'walker2d_full_replay-v2'
    data = read_hdf5_files(task)

    discount = 0.99
    count = 0
    start = 0
    end = 0
    returns = np.zeros_like(data['rewards'])
    done = np.zeros_like(data['terminals'])
    for i in range(len(data['rewards'])):
        timeout = data['timeouts'][i]
        terminal = data['terminals'][i]
        if timeout == True or terminal == True:
            end = i
            done[i] = 1
            # traj_reward = data['rewards'][start:end]
            # discounts = discount ** np.arange(len(traj_reward))?
            return_ = 0
            for j in range(end, start - 1, -1):
                return_ = data['rewards'][j] + discount * return_
                returns[j] = return_

            start = end + 1

    save_path = './dataset/datasets/'+ task
    np.savez(save_path,
             obs=data['observations'],
             action=data['actions'],
             next_obs=data['next_observations'],
             reward=data['rewards'],
             returns=returns,
             done=done
             )

    print('ok')
