import torch
import random
import numpy as np

from torchvision import datasets, transforms


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data_by_id(dataset_name, use_train_data=False, data_path=None):
    assert dataset_name in ['cifar10', 'imagenet', 'mnist']
    assert data_path is not None
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        # transform = None
        test_set = datasets.CIFAR10(
            root=data_path,
            train=use_train_data, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.Resize(232),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
        test_set = datasets.ImageNet(
            root=data_path,
            split='val' if not use_train_data else 'train', transform=transform)
    elif dataset_name == 'mnist':
        # transform = transforms.Compose([transforms.ToTensor()])
        transform = None
        test_set = datasets.MNIST(
            root=data_path, train=use_train_data, download=True,
            transform=transform)
    else:
        raise NotImplementedError("Dataset not supported")

    return test_set


# The code for creating a subset of the dataset (1000 images)
def create_idx_by_id(dataset_name, data_set, save_path=None):
    if dataset_name == 'cifar10':
        unique_label_list = [i for i in range(10)]
        label_idx_list = []
        for each_unique_label in unique_label_list:
            idx = np.where(np.array(data_set.targets) == each_unique_label)[0]
            rand_perm = np.random.permutation(idx.shape[0])[0:100]
            idx = idx[rand_perm]
            label_idx_list.append(idx)
        subset_idx = np.stack(label_idx_list)
        subset_idx = subset_idx.reshape(-1)
        subset_idx = subset_idx.astype(int)
        np.savetxt(save_path, subset_idx, fmt='%i')
    elif dataset_name == 'imagenet':
        subset_idx = np.random.permutation(len(data_set))[0:1000]
        np.savetxt(save_path, subset_idx, fmt='%i')
