import torch
from torch import nn
from torchvision.models.vgg import vgg16

from physics_loss import PhysicsInformedLoss


class GeneratorLoss(nn.Module):
    def __init__(self, loss_weight, image_loss_weight, pi_params, lambda_params, device):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_weight = loss_weight
        self.image_loss_weight = image_loss_weight
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()
        self.pi_loss = PhysicsInformedLoss(*lambda_params, *pi_params, device)

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(
            out_images), self.loss_network(target_images))
        # Image Loss
        # image_loss = self.image_loss_weight[0] * self.mse_loss(out_images[:, 0, :, :], target_images[:, 0, :, :]) + \
        #     self.image_loss_weight[1] * self.mse_loss(out_images[:, 1, :, :], target_images[:, 1, :, :]) + \
        #     self.image_loss_weight[2] * self.mse_loss(
        #         out_images[:, 2, :, :], target_images[:, 2, :, :])
        image_loss = (self.image_loss_weight[0] * self.mse_loss(out_images[:, 0, :, :], target_images[:, 0, :, :])) / (torch.max(target_images[:, 0, :, :]) ** 2) + \
                     (self.image_loss_weight[1] * self.mse_loss(out_images[:, 1, :, :], target_images[:, 1, :, :])) / (torch.max(target_images[:, 1, :, :]) ** 2) + \
                     (self.image_loss_weight[2] * self.mse_loss(out_images[:, 2, :, :], target_images[:, 2, :, :])) / (torch.max(target_images[:, 2, :, :]) ** 2)

        image_loss = image_loss / 3
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        # PI loss added 20201220
        pi_loss = self.pi_loss(out_images)
        return self.loss_weight[0] * image_loss,\
            self.loss_weight[1] * adversarial_loss,\
            self.loss_weight[2] * perception_loss,\
            self.loss_weight[3] * tv_loss,\
            self.loss_weight[4] * pi_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
