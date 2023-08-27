import torch
import torch.nn as nn
import torchvision


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gpu_ids, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        target_real_label = torch.tensor(target_real_label)
        target_fake_label = torch.tensor(target_fake_label)
        if len(gpu_ids) > 0:
            target_real_label = target_real_label.cuda()
            target_fake_label = target_fake_label.cuda()
        self.register_buffer('real_label', target_real_label)
        self.register_buffer('fake_label', target_fake_label)
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


'''
define the correlation coefficient loss
'''


def Cor_CoeLoss(y_pred, y_target):
    x = y_pred
    y = y_target
    x_var = x - torch.mean(x)
    y_var = y - torch.mean(y)
    r_num = torch.sum(x_var * y_var)
    r_den = torch.sqrt(torch.sum(x_var ** 2)) * torch.sqrt(torch.sum(y_var ** 2))
    r = r_num / r_den

    # return 1 - r  # best are 0
    return 1 - r ** 2  # abslute constrain


'''
define the percepetual loss (2D-wise)
'''


class PerceptualLoss(nn.Module):

    def __init__(self, gpu_ids, f1=0.5, f2=0.5, f3=0.5):
        vgg19 = torchvision.models.vgg19(pretrained=True)
        first_features = nn.Sequential(*list(vgg19.children())[0][:6])
        second_features = nn.Sequential(*list(vgg19.children())[0][6:15])
        third_features = nn.Sequential(*list(vgg19.children())[0][15:26])
        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            if len(gpu_ids) > 1:
                # 3 layer
                first_features = torch.nn.DataParallel(first_features)
                # 7 layer
                second_features = torch.nn.DataParallel(second_features)
                # 12 layer
                third_features = torch.nn.DataParallel(third_features)
            first_features.cuda()
            second_features.cuda()
            third_features.cuda()

        first_features.eval()
        second_features.eval()
        third_features.eval()
        first_features.requires_grad_(False)
        second_features.requires_grad_(False)
        third_features.requires_grad_(False)
        self.feature_extractors = [first_features, second_features, third_features]

        self.feature_weights = [f1, f2, f3]

    def __call__(self, A_reconstructed, A):
        loss = 0.
        for i in range(A.shape[2]):
            A_f = torch.cat((A[:, :, i, :, :], A[:, :, i, :, :], A[:, :, i, :, :]), 1)
            A_reconstructed_f = torch.cat(
                (A_reconstructed[:, :, i, :, :], A_reconstructed[:, :, i, :, :], A_reconstructed[:, :, i, :, :]), 1)
            # 3_layer
            A_f1 = self.feature_extractors[0](A_f)
            A_reconstructed_f1 = self.feature_extractors[0](A_reconstructed_f)
            loss += self.feature_weights[0] * (torch.mean(torch.abs(A_f1 - A_reconstructed_f1)))
            # 7 layer
            A_f2 = self.feature_extractors[1](A_f1)
            A_reconstructed_f2 = self.feature_extractors[1](A_reconstructed_f1)
            loss += self.feature_weights[1] * (torch.mean(torch.abs(A_f2 - A_reconstructed_f2)))
            # 12 layer
            A_f3 = self.feature_extractors[2](A_f2)
            A_reconstructed_f3 = self.feature_extractors[2](A_reconstructed_f2)
            loss += self.feature_weights[2] * (torch.mean(torch.abs(A_f3 - A_reconstructed_f3)))
        loss /= (10 * A.shape[2])
        return loss
