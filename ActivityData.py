# Create input pipeline
import numpy as np
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import RandomSampler
from scipy.spatial.transform import Rotation
import copy


# Define dataset
class ActivityDataset(Dataset):
    def __init__(self, root_dir, window_length=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels2idx = {"no action": 0, "go_to": 1, "pick": 2}
        self.data = []
        # not sure if this mapping is correct
        self.object2idx = {'HandRight': 0, 'tomato': 1, 'dish': 2, 'glass': 3, 'Tea': 4}
        for filename in glob.iglob('%s/*.npy' % root_dir):
            trajectories = np.load(filename)
            # obtain labels from filename
            label = filename.split("/")[-1][:-4]
            label_list = label.split("_")[:-1]
            object_label = label.split("_")[-1]
            activity_label = "_".join(label_list)
            # we select only a predefined subset of labels
            if activity_label not in self.labels2idx.keys():
                continue
            # select only a time segment for input data
            windows = rolling_window(trajectories, window_length)
            for window in windows:
                self.data.append(dict(trajectory=window, activity_label=activity_label,
                                      object_label=object_label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample


""" 
since traindata and test data have different structure, the easiest I have come up with to capture this is with
a separate classes
"""


class ActivityTrainSet(Dataset):
    def __init__(self, activity_data, indices):
        # indices are a random percentage of indices of the original dataset
        self.labels2idx = activity_data.labels2idx
        self.object2idx = activity_data.object2idx
        self.transform = activity_data.transform
        self.data = []
        for index in indices:
            activity_label = activity_data[index]["activity_label"]
            object_label = activity_data[index]["object_label"]
            trajectory = activity_data[index]["trajectory"]
            self.data.append(dict(trajectory=trajectory[:, self.object2idx[object_label], :]
                                  , label=self.labels2idx[activity_label]))
            for obj, idx in self.object2idx.items():
                if obj in ("HandRight", object_label):
                    continue  # skip hand and object involved
                # label trajectories of objects that are non involved with "no action"
                self.data.append(dict(trajectory=trajectory[:, [0, idx], :], label=self.labels2idx["no action"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.data[idx])
        if self.transform:
            sample = self.transform(sample)
        return sample


class ActivityTestSet(Dataset):
    def __init__(self, activity_data, indices):
        # indices are a random percentage of indices of the original dataset
        self.labels2idx = activity_data.labels2idx
        self.object2idx = activity_data.object2idx
        self.transform = activity_data.transform
        self.data = []
        for index in indices:
            activity_label = activity_data[index]["activity_label"]
            trajectory = activity_data[index]["trajectory"]
            object_label = activity_data[index]["object_label"]
            self.data.append(activity_data[index])
            self.data.append(dict(trajectory=trajectory,
                                  activity_label=self.labels2idx[activity_label],
                                  object_label=self.object2idx[object_label]))

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
def train_dev_test_loader(activity_data, split={'train': 0.8, "dev": 0.1, 'test': 0.1}):
    np.random.seed(420)
    # first we split the test set from the rest
    n = len(activity_data)
    indices = list(range(n))
    np.random.shuffle(indices)
    split_train = int(split['train'] * n)
    rest_idx, test_idx = indices[:split_train], indices[split_train:]
    train_data = ActivityTrainSet(act_dataset, rest_idx)
    test_data = ActivityTestSet(act_dataset, test_idx)
    # then we split train and dev
    m = len(train_data)
    rest_indices = list(range(m))
    # since the total index size is now reduced to train split, we have to rescale the dev split
    split_dev = int(split['dev'] * m * 1 / split["train"])
    dev_idx, train_idx = rest_indices[:split_dev], rest_indices[split_dev:]
    train_sampler = SubsetRandomSampler(train_idx)
    dev_sampler = SubsetRandomSampler(dev_idx)
    test_sampler = RandomSampler(test_data)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=1)
    devloader = torch.utils.data.DataLoader(train_data,
                                            sampler=dev_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data,
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
    act_dataset = ActivityDataset('./Data/27_04', window_length=5)
    trainloader, devloader, testloader = train_dev_test_loader(act_dataset)
    for i, sample in enumerate(trainloader):
        trajectory = sample["trajectory"]
        label = sample["label"]
        print(label)
        print(trajectory.shape)
        if i == 0:
            break

    for i, sample in enumerate(testloader):
        trajectory = sample["trajectory"]
        activity_label = sample["activity_label"]
        object_label = sample["object_label"]
        print(activity_label)
        print(object_label)
        print(trajectory.shape)
        if i == 0:
            break
