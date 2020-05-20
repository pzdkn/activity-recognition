# Create input pipeline
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.spatial.transform import Rotation
import copy


# Define dataset
class ActivityDataset(Dataset):
    def __init__(self, root_dir, window_length=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = {}
        self.data = []
        # not sure if this mapping is correct
        self.object2idx = {'HandRight': 0, 'tomato': 1, 'dish': 2, 'glass': 3, 'Tea': 4}
        for label_idx, filename in enumerate(glob.iglob('%s/*.npy' % root_dir)):
            trajectories = np.load(filename)
            label = filename.split("/")[-1][:-4]
            # right now I just use the last object in the label name as object in quesiton. However this should be
            # changed TODO!: extend to multiple object/multiple actions -> multilabel classification ?
            object_label = label.split("_")[-1]
            activity_label = "_".join(label.split("_")[:-1])
            self.labels[label_idx] = activity_label
            windows = rolling_window(trajectories, window_length)
            for window in windows:
                trajectory = window[:, [0, self.object2idx[object_label]], :]
                self.data.append(dict(trajectory=trajectory, label=label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample

# extract a rolling time window from the signals
def rolling_window(trajectories, window_length):
    for trajectory in trajectories:
        trajectory = np.array(trajectory, dtype=np.float64)
        window = trajectory[:window_length]
        length = trajectory.shape[0]
        yield window
        for i in range(window_length, length):
            window[:-1] = window[1:]
            window[-1] = trajectory[i]
            yield window


# shuffle the data, and define the split
def train_dev_test_loader(data, split={'train': 0.8, 'dev': 0.1}):
    np.random.seed(420)
    n = len(data)
    indices = list(range(n))
    np.random.shuffle(indices)
    split0 = int(split['train'] * n)
    split1 = int(split['dev'] * n)
    train_idx, dev_idx, test_idx = indices[:split0], indices[split0:split0 + split1], indices[split0 + split1:]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(data,
                                              sampler=train_sampler, batch_size=1)
    devloader = torch.utils.data.DataLoader(data,
                                            sampler=dev_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(data,
                                             sampler=test_sampler, batch_size=1)
    return trainloader, devloader, testloader

# this is used to get a homogenous transformation from object frame to human frame
def homogenous_trans(sample):
    # get the matrix
    def get_trans_matrix(x, y, z, a, b, c, d):
        r = Rotation.from_quat([a, b, c, d])
        R = r.as_matrix()
        T = np.concatenate((R.transpose(), -R.transpose().dot(np.array([[x], [y], [z]]))), axis=1)
        z = np.zeros((1, 4))
        z[0, -1] = 1
        T = np.concatenate((T, z), axis=0)
        return T

    # do homogenous transform
    trajectory = sample["trajectory"]
    for i in range(trajectory.shape[0]):
        T = get_trans_matrix(*trajectory[i, 0, :])
        t_h = np.concatenate((trajectory[i, 1, :3], np.array([1])), axis=0)
        trajectory[i, 1, :3] = T.dot(t_h)[:3]
    return sample

# a cheap transformation to make object position relative to human position
def pseudo_relative_trans(sample):
    trajectory = sample["trajectory"]
    # subtracting human position from object position
    trajectory[:, 0, :3] -= trajectory[:, 1, :3]
    # playing with python's copy by reference
    return sample

# toy transform to play with
def pseudo_toy_trans(sample):
    sample["trajectory"] *= 100
    return sample

if __name__ == '__main__':
    # Create dataset & loader
    act_dataset = ActivityDataset('../Data/27_04', window_length=5, transform=pseudo_relative_trans)
    trainloader, devloader, testloader = train_dev_test_loader(act_dataset)
    for data in trainloader:
        print(data)
        tr = data["trajectory"]
        print(tr.shape)
        label = data["label"]
        print(label)
        break
