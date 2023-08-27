import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from models import create_model
from options.train_options import TrainOptions
from utils.UnpairedDataset import UnpairedDataset
from utils.earlystop import EarlyStopping
from utils.visualizer import Visualizer


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    # [option] to seed the seed
    # seed_torch(opt.seed)

    # -----  Transformation and Augmentation process for the data  -----

    train_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="train", load_size=opt.load_size,
                                crop_size=opt.crop_size)
    valid_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="valid", load_size=opt.load_size,
                                crop_size=opt.crop_size)
    print('length train list:', len(train_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=True)  # Here are then fed to the network with a defined batch size
    valid_loader = DataLoader(valid_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers,
                              pin_memory=False)  # Here are then fed to the network with a defined batch size

    # initialize the early_stopping object
    if opt.use_earlystop:
        print('using early stop')
        early_stopping = EarlyStopping(patience=opt.patience, verbose=True)

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    visualizer = Visualizer(opt, train_loader)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            if total_steps % opt.update_step == 0:
                model.optimize_parameters(opt.update_step, True)
            else:
                model.optimize_parameters(opt.update_step, False)
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            if total_steps % opt.eval_freq == 0:
                loss_G_list = []
                with torch.no_grad():
                    model.eval()  # prep model for evaluation
                    for i, data in enumerate(valid_loader):
                        # forward pass: compute predicted outputs by passing inputs to the model
                        model.set_input(data)
                        loss_G_list.append(model.get_current_losses()['G'])

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                if opt.use_earlystop:
                    early_stopping(np.mean(loss_G_list), model, epoch)
                    if early_stopping.early_stop:
                        print("Early stopping from iteration")
                        break
        if opt.use_earlystop:
            if early_stopping.early_stop:
                print("Early stopping from epoch")
                break

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
