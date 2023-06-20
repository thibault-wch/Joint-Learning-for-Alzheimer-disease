import os
import time

import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


class Visualizer():
    def __init__(self, opt, train_loader):
        self.name = opt.name
        self.opt = opt
        self.batch_size = opt.batch_size
        self.saved = False
        self.init_epoch = opt.epoch_count - 1
        self.first_flag = 0
        self.iters_total = (opt.epoch_count) * (len(train_loader) * opt.batch_size)
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        self.log_tb = os.path.join(opt.checkpoints_dir, opt.name, 'tblogs')
        self.lossprintiter = LossPrintIter()
        if not os.path.exists(self.log_tb):
            os.makedirs(self.log_tb)
        self.tbwriter = SummaryWriter(log_dir=self.log_tb)

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        # empty the print buffer
        if epoch > self.init_epoch:
            if self.first_flag == 0:
                self.first_flag += 1
            else:
                message = '[epoch summary]: %d) ' % (self.init_epoch)
                for key in self.lossprintiter.loss_buffer:
                    message += '%s: %.3f ' % (key + '_epoch', self.lossprintiter.return_value(key))
                    self.tbwriter.add_scalar('{}'.format(key + '_epoch'), self.lossprintiter.return_value(key),
                                             self.init_epoch)
                print(message)
                with open(self.log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
            self.init_epoch += 1
            self.lossprintiter.buffer_empty()

        self.iters_total += self.batch_size
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, i, t, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            self.lossprintiter.add_value(k, v)
            self.tbwriter.add_scalar('{}'.format(k), v, self.iters_total)
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
        return


class LossPrintIter():
    def __init__(self):
        self.loss_name = ['D_A', 'G', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'Gen', 'Cls']
        self.loss_buffer = {}

    def buffer_empty(self):
        for item in self.loss_name:
            self.loss_buffer[item] = []
        return

    def add_value(self, k, v):
        self.loss_buffer[k].append(v)
        return

    def return_value(self, k):
        return np.mean(self.loss_buffer[k])


# only for diagnosis network plot
def plt_result(dataframe, checkpoint_path):
    fig = plt.figure(figsize=(20, 12))

    fig.add_subplot(2, 3, 1)
    plt.plot(dataframe['epoch'], dataframe['train_loss'], 'bo', label='Train loss')
    plt.plot(dataframe['epoch'], dataframe['val_loss'], 'b', label='Val loss')
    plt.title('Training and validation loss')
    plt.legend()

    fig.add_subplot(2, 3, 2)
    plt.plot(dataframe['epoch'], dataframe['train_acc'], 'bo', label='Train Accuracy')
    plt.plot(dataframe['epoch'], dataframe['val_acc'], 'b', label='Val Accuracy')
    plt.title('Training and validation Accuracy')
    plt.legend()

    fig.add_subplot(2, 3, 3)
    plt.plot(dataframe['epoch'], dataframe['train_f1_score'], 'bo', label='Train F1 Score')
    plt.plot(dataframe['epoch'], dataframe['val_f1_score'], 'b', label='Val F1 Score')
    plt.title('Training and validation F1 Score')
    plt.legend()

    fig.add_subplot(2, 3, 4)
    plt.plot(dataframe['epoch'], dataframe['train_spe'], 'bo', label='Train Specificity')
    plt.plot(dataframe['epoch'], dataframe['val_spe'], 'b', label='Val Specificity')
    plt.title('Training and validation Specificity')
    plt.legend()

    fig.add_subplot(2, 3, 5)
    plt.plot(dataframe['epoch'], dataframe['train_recall'], 'bo', label='Train Sensitivity')
    plt.plot(dataframe['epoch'], dataframe['val_recall'], 'b', label='Val Sensitivity')
    plt.title('Training and validation Sensitivity')
    plt.legend()

    fig.add_subplot(2, 3, 6)
    plt.plot(dataframe['epoch'], dataframe['train_auc'], 'bo', label='Train AUC')
    plt.plot(dataframe['epoch'], dataframe['val_auc'], 'b', label='Val AUC')
    plt.title('Training and validation AUC')
    plt.legend()

    plt.savefig('{}/performance_metric_summary.png'.format(checkpoint_path))
    plt.show()
