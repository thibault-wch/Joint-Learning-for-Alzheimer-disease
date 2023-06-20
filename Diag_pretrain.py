import argparse
import os

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from components import networks3D
from utils.Diag_pretraining import train_data
from utils.UnpairedDataset import UnpairedDataset
from utils.utils import mkdir

if __name__ == '__main__':
    # args definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', default='7', help='gpu ids: e.g. 0')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--gamma', type=float, default=0.9, help='basic gamma value for exponentialLR')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--use_early_stop', action='store_true', help='use early stop')
    parser.add_argument('--patience', type=int, default=5,
                        help='How long to wait after last time validation loss improved.')
    parser.add_argument('--checkpoints_dir', type=str, default='/data/chwang/Log/ShareGAN',
                        help='models are saved here')
    parser.add_argument('--name', type=str, default='DiagNet', help='saving name')
    parser.add_argument('--load_size', default=256, help='Size of the original image')
    parser.add_argument('--crop_size', default=128, help='Size of the patches extracted from the image')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    model = networks3D.define_Cls(2, args.init_type, args.init_gain, args.gpu_ids)
    epochs = args.n_epochs
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
    criterion = nn.CrossEntropyLoss()
    train_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="train", load_size=args.load_size,
                                crop_size=args.crop_size)
    valid_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="valid", load_size=args.load_size,
                                crop_size=args.crop_size)
    print('length train list:', len(train_set))
    print('length valid list:', len(valid_set))
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=False)
    save_dir = args.checkpoints_dir + '/' + args.name + '/'
    mkdir(save_dir)
    train_data(model, train_loader, valid_loader, epochs, optimizer, scheduler, criterion, args.use_early_stop,
               args.patience, args.gpu_ids, save_dir, args.save_freq)
