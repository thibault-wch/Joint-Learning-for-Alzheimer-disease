import pickle

import numpy as np
import torch

from .utils import RandomCrop3D


class UnpairedDataset(torch.utils.data.Dataset):
    """
    Dataset class

    Parameters:
        root - - : (str) Path of the root folder
        mode - - : (str) {'train' or 'test'} Part of the stage
        seed - - : (int) random seed
        dataset - - : (str) {'adni', 'aibl' or 'nacc'} Part of the dataset
    """

    def __init__(self, data_list=['0', '1'],
                 which_direction='AtoB',
                 mode="train",
                 dataset="adni",
                 load_size=256,
                 crop_size=256,
                 ):
        # basic initialize
        self.mode = mode
        self.data_list = data_list
        self.which_direction = which_direction
        self.load_size = load_size
        self.crop_size = crop_size
        self.dataset = dataset

        # [Important note]
        # Note that the PET images is unpaired for the internal ADNI training set, while is paired for the internal ADNI validata and the testing set;
        # The training pairs are randomly pre-setted for better examination, but the pair number is five times the subject number possessing MRI images in the training set.

        # training / validation set
        self.train_pairs = pickle.load(open('/data/chwang/final_dataset_MRI/lineared/final_train_pairs.pkl', 'rb'))
        self.valid_pairs = pickle.load(open('/data/chwang/final_dataset_MRI/lineared/final_valid_pairs.pkl', 'rb'))

        if dataset == "adni":
            # internal testing set
            self.test_pairs = pickle.load(open('/data/chwang/final_dataset_MRI/lineared/final_test_pairs.pkl', 'rb'))
        elif dataset == "aibl":
            # external aibl testing set
            self.test_pairs = pickle.load(open('/data/chwang/final_dataset_MRI/AIBL/aibl_datapair.pkl', 'rb'))
        elif dataset == "nacc":
            # external nacc testing set
            self.test_pairs = pickle.load(open('/data/chwang/final_dataset_MRI/NACC/nacc_datapair.pkl', 'rb'))

        self.imgs = []
        if self.load_size == self.crop_size:
            self.is_transform = False
        else:
            self.is_transform = True
            self.transforms = RandomCrop3D((1, int(self.load_size), int(self.load_size), int(self.load_size)),
                                           (int(self.crop_size), int(self.crop_size), int(self.crop_size)))

        if mode == "train":
                self.imgs = self.train_pairs
        elif mode == "test":
                self.imgs = self.test_pairs
        elif mode == "valid":
                self.imgs = self.valid_pairs

    def __getitem__(self, index):
        # extract the MRI images
        data_path = self.imgs[index][0]
        # wheather to extract the PET images
        if self.dataset == "adni":
            label_path = self.imgs[index][1]
            if self.which_direction == 'AtoB':
                label_path = label_path
            elif self.which_direction == "BtoA":
                tmp = data_path
                data_path = label_path
                label_path = tmp
            image_np = np.pad(np.load(data_path), ((37, 37), (19, 19), (37, 37)), 'constant', constant_values=(-1, -1))
            A = torch.from_numpy(image_np).unsqueeze(0).type(torch.FloatTensor)
            label_np = np.pad(np.load(label_path), ((37, 37), (19, 19), (37, 37)), 'constant', constant_values=(-1, -1))
            B = torch.from_numpy(label_np).unsqueeze(0).type(torch.FloatTensor)
            if self.is_transform == True:
                A = self.transforms(A, slice_change=True)
                B = self.transforms(B, slice_change=False)
            value = self.imgs[index][2]
        # for aibl and nacc dataset
        else:
            image_np = np.pad(np.load(data_path), ((37, 37), (19, 19), (37, 37)), 'constant', constant_values=(-1, -1))
            A = torch.from_numpy(image_np).unsqueeze(0).type(torch.FloatTensor)
            B = -1
            value = self.imgs[index][1]
        return A, B, value

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)
