import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from components import networks3D
from components.performance_metric import mean_absolute_error, peak_signal_to_noise_ratio, structural_similarity_index
from options.test_options import TestOptions
from utils.UnpairedDataset import UnpairedDataset


def evaluate_generator(generator, test_loader, netG):
    """Evaluate a generator.

    Parameters:
        generator - - : (nn.Module) neural network generating PET images
        train_loader - - : (dataloader) the training loader
        test_loader - - : (dataloader) the testing loader

    Returns:
        df - - : (dataframe) the dataframe of the different Sets
    """
    res_test = []

    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    generator.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            # Inputs MRI and PET
            real_mri = batch[0].type(Tensor)
            real_pet = batch[1].type(Tensor)
            if netG == 'ShareSynNet':
                fake_pet = generator(real_mri, alpha=0.)
            else:
                fake_pet = generator(real_mri)
            mae = mean_absolute_error(real_pet, fake_pet).item()
            psnr = peak_signal_to_noise_ratio(real_pet, fake_pet).item()
            ssim = structural_similarity_index(real_pet, fake_pet).item()
            res_test.append([mae, psnr, ssim])

        df = pd.DataFrame([
            pd.DataFrame(res_test, columns=['MAE', 'PSNR', 'SSIM']).mean().squeeze()
        ], index=['Test set']).T
    return df


if __name__ == '__main__':
    opt = TestOptions().parse()

    netG = networks3D.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                               not opt.no_dropout, opt.init_type, opt.init_gain, opt.gpu_ids)
    print('netG:\ttype:{}\tload_path:{}'.format(opt.netG, opt.load_path))
    test_set = UnpairedDataset(data_list=['0', '1'], which_direction='AtoB', mode="test", load_size=opt.load_size,
                               crop_size=opt.crop_size)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=opt.workers,
                             pin_memory=True)  # Here are then fed to the network with a defined batch siz

    print('lenght test list:', len(test_set))

    state_dict = torch.load(opt.load_path)
    netG.load_state_dict(state_dict)
    test_df = evaluate_generator(netG, test_loader, opt.netG)
    print(test_df)
