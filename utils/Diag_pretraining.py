import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from tqdm import tqdm

from .visualizer import plt_result


def train_data(model, train_dataloaders, valid_dataloaders, epochs, optimizer, scheduler, criterion, use_early_stop,
               patience, gpu_ids, checkpoints_dir, save_freq):
    '''
        model: the diagnosis model
        train_dataloaders: the training dataloader
        epochs: training epoch number
        optimizer: optimizer
        scheduler: learning rate scheduler
        criterion: loss function criterion
        use_early_stop : wheather to use early stopping
        patience : the number of early stopping iteration
        gpu_ids : gpu_ids
        checkpoint_path: the checkppoint path
    '''
    start = time.time()
    if use_early_stop:
        print('Start using early stopping!')
    model_indicators = pd.DataFrame(
        columns=['epoch', 'train_loss', 'train_acc', 'train_f1_score', 'val_loss', 'val_acc', 'val_f1_score',
                 'train_recall', 'val_recall', 'train_spe', 'val_spe', 'train_auc', 'val_auc'])
    steps = 0
    epochs_no_improve = 0
    min_val_loss = np.Inf
    for e in range(epochs):

        model.train()
        train_loss = 0
        train_correct_sum = 0
        train_simple_cnt = 0
        y_train_true = []
        y_train_pred = []
        train_prob_all = []
        train_label_all = []
        for ii, (images, _, labels) in enumerate(tqdm(train_dataloaders)):
            steps += 1
            images, labels = images.squeeze(1).cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs, _, _, _, _ = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct_sum += (labels.data == train_predicted).sum().item()
            train_simple_cnt += labels.size(0)
            y_train_true.extend(np.ravel(np.squeeze(labels.cpu().detach().numpy())).tolist())
            y_train_pred.extend(np.ravel(np.squeeze(train_predicted.cpu().detach().numpy())).tolist())
            train_prob_all.extend(outputs[:,
                                  1].cpu().detach().numpy())
            train_label_all.extend(labels.cpu())

        scheduler.step()
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
                val_prob_all.extend(outputs[:,
                                    1].cpu().detach().numpy())
                val_label_all.extend(labels.cpu())

        train_loss = train_loss / len(train_dataloaders)
        val_loss = val_loss / len(valid_dataloaders)
        train_acc = train_correct_sum / train_simple_cnt
        val_acc = val_correct_sum / val_simple_cnt
        val_f1_score = f1_score(y_val_true, y_val_pred, average='weighted')
        val_recall = recall_score(y_val_true, y_val_pred, average='weighted')
        val_auc = roc_auc_score(val_label_all, val_prob_all, average='weighted')
        train_f1_score = f1_score(y_train_true, y_train_pred, average='weighted')
        train_recall = recall_score(y_train_true, y_train_pred, average='weighted')
        val_spe = recall_score(y_val_true, y_val_pred, pos_label=0, average='binary')
        train_spe = recall_score(y_train_true, y_train_pred, pos_label=0, average='binary')
        train_auc = roc_auc_score(train_label_all, train_prob_all, average='weighted')

        print('Epochs: {}/{}...'.format(e + 1, epochs),
              'Trian Loss:{:.3f}...'.format(train_loss),
              'Trian Accuracy:{:.3f}...'.format(train_acc),
              'Trian AUC:{:.3f}...'.format(train_auc),
              'Trian F1 Score:{:.3f}...'.format(train_f1_score),
              'Trian SPE:{:.3f}...'.format(train_spe),
              'Trian SEN:{:.3f}...'.format(train_recall),
              'Val Loss:{:.3f}...'.format(val_loss),
              'Val Accuracy:{:.3f}...'.format(val_acc),
              'Val AUC:{:.3f}...'.format(val_auc),
              'Val F1 Score:{:.3f}'.format(val_f1_score),
              'val SPE:{:.3f}...'.format(val_spe),
              'Val SEN:{:.3f}...'.format(val_recall)
              )
        model_indicators.loc[model_indicators.shape[0]] = [e, train_loss, train_acc, train_f1_score, val_loss, val_acc,
                                                           val_f1_score, train_recall, val_recall, train_spe, val_spe,
                                                           train_auc, val_auc]

        if e % save_freq == 0:
            if len(gpu_ids) > 1 and torch.cuda.is_available():
                torch.save(model.module.state_dict(), '{}/{}_Cls.pth'.format(checkpoints_dir, e))
            else:
                torch.save(model.state_dict(), '{}/{}_Cls.pth'.format(checkpoints_dir, e))
        # early stopping
        if use_early_stop:
            if val_loss < min_val_loss:
                print(f'Validation loss decreased ({min_val_loss:.4f} --> {val_loss:.4f}).  Saving model ...')
                if len(gpu_ids) > 1 and torch.cuda.is_available():
                    torch.save(model.module.state_dict(), '{}/{}_{:.4f}_Cls.pth'.format(checkpoints_dir, e, val_loss))
                else:
                    torch.save(model.state_dict(), '{}/{}_{:.4f}_Cls.pth'.format(checkpoints_dir, e, val_loss))
                min_val_loss = val_loss
            else:
                epochs_no_improve += 1
                print(f'EarlyStopping counter: {epochs_no_improve} out of {patience}')
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    break

    plt_result(model_indicators, checkpoints_dir)
    end = time.time()
    runing_time = end - start
    print('Training time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
