import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from components import networks3D
from utils.UnpairedDataset import UnpairedDataset


def evaluate_diagNetwork(model, valid_dataloaders):
    """Evaluate a generator.

    Parameters:
        generator - - : (nn.Module) neural network generating PET images
        train_loader - - : (dataloader) the training loader
        test_loader - - : (dataloader) the testing loader

    Returns:
        df - - : (dataframe) the dataframe of the different Sets
    """
    criterion = nn.CrossEntropyLoss()
    val_correct_sum = 0
    val_simple_cnt = 0
    val_loss = 0
    y_val_true = []
    y_val_pred = []
    val_prob_all = []
    val_label_all = []
    with torch.no_grad():
        model.eval()
        for ii, (images, _, labels) in enumerate(tqdm(valid_dataloaders)):
            images, labels = images.squeeze(1).cuda(), labels.cuda()
            outputs, _, _, _, _ = model(images)
            val_loss += criterion(outputs, labels).item()
            _, val_predicted = torch.max(outputs.data, 1)
            val_correct_sum += (labels.data == val_predicted).sum().item()
            val_simple_cnt += labels.size(0)
            y_val_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_val_pred.extend(np.ravel(np.squeeze(val_predicted.cpu().detach().numpy())).tolist())
            outputs=outputs.softmax(dim=-1)
            val_prob_all.extend(outputs[:,
                                1].cpu().detach().numpy())
            val_label_all.extend(labels.cpu())

    val_loss = val_loss / len(valid_dataloaders)
    val_acc = val_correct_sum / val_simple_cnt
    val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
    val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
    val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
    val_auc = roc_auc_score(val_label_all, val_prob_all, average='weighted')

    print(
        'Val Loss:{:.3f}...'.format(val_loss),
        'Val Accuracy:{:.3f}...'.format(val_acc),
        'Val AUC:{:.3f}...'.format(val_auc),
        'Val F1 Score:{:.3f}'.format(val_f1_score),
        'val SPE:{:.3f}...'.format(val_spe),
        'Val SEN:{:.3f}...'.format(val_recall)
    )


if __name__ == '__main__':
    # args definition
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', default='7', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--init_type', type=str, default='normal',
                        help='network initialization [normal|xavier|kaiming|orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02,
                        help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--load_path', type=str, default='/data/chwang/Log/ShareGAN/Cls.pth',
                        help='models are saved here')
    parser.add_argument('--load_size', default=256, help='Size of the original image')
    parser.add_argument('--crop_size', default=128, help='Size of the patches extracted from the image')
    parser.add_argument('--dataset', default="adni", type=str, help='Types of dataset [adni|aibl|nacc]')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # test set
    test_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="test", load_size=args.load_size,
                               crop_size=args.crop_size, dataset=args.dataset)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=args.workers,
                             pin_memory=True)  # Here are then fed to the network with a defined batch size
    print('length test list:', len(test_set))
    # model definition
    print('initialize the model')
    model = networks3D.define_Cls(2, args.init_type, args.init_gain, args.gpu_ids)
    print('loading state dict from : {}'.format(args.load_path))
    state_dict = torch.load(args.load_path, map_location='cuda')
    model.load_state_dict(state_dict)
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        if len(args.gpu_ids) > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()

    # model evaluation
    evaluate_diagNetwork(model, test_loader)
