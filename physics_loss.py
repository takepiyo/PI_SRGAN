import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):

    def __init__(self, labmda_con, lambda_b_c, dx, dt, u_0, visc, crop_size, device='cpu'):
        super(PhysicsInformedLoss, self).__init__()
        self.labmda_con = labmda_con
        self.lambda_b_c = lambda_b_c
        self.dx = dx.to(device)
        self.dt = dt.to(device)
        self.u_0 = u_0.to(device)
        self.crop_size = crop_size
        self.N = crop_size - 2
        self.visc = visc.to(device)
        self.div_ddx = nn.Conv2d(1, 1, 3, 1, 0, bias=False).to(device)
        self.div_ddy = nn.Conv2d(1, 1, 3, 1, 0, bias=False).to(device)
        self.average_x = nn.Conv2d(1, 1, 2, 1, 0, bias=False).to(device)
        self.average_y = nn.Conv2d(1, 1, 2, 1, 0, bias=False).to(device)
        self.poisson_up_ddx = nn.Conv2d(1, 1, 2, 1, 0, bias=False).to(device)
        self.poisson_up_ddy = nn.Conv2d(1, 1, 2, 1, 0, bias=False).to(device)
        self.poisson_down_ddx = nn.Conv2d(
            1, 1, 2, 1, 0, bias=False).to(device)
        self.poisson_down_ddy = nn.Conv2d(
            1, 1, 2, 1, 0, bias=False).to(device)
        self.laplacoan_filter = nn.Conv2d(
            1, 1, 3, 1, 0, bias=False).to(device)
        self.device = device
        self.setting_weights()

    def setting_weights(self):
        self.div_ddx.weight.data = torch.tensor([[[[0, 0, 0],
                                                   [-1., 1., 0],
                                                   [0, 0, 0]]]], device=self.device)

        self.div_ddy.weight.data = torch.tensor([[[[0, -1., 0],
                                                   [0, 1, 0],
                                                   [0, 0, 0]]]], device=self.device)

        self.average_x.weight.data = torch.tensor([[[[0.5, 0.5],
                                                     [0., 0.]]]], device=self.device)

        self.average_y.weight.data = torch.tensor([[[[0.5, 0.],
                                                     [0.5, 0.]]]], device=self.device)

        self.poisson_up_ddx.weight.data = torch.tensor([[[[-1., 1.],
                                                          [0, 0]]]], device=self.device)

        self.poisson_up_ddy.weight.data = torch.tensor([[[[-1., 0],
                                                          [1., 0]]]], device=self.device)

        self.poisson_down_ddx.weight.data = torch.tensor([[[[0., 0],
                                                            [-1., 1.]]]], device=self.device)

        self.poisson_down_ddy.weight.data = torch.tensor([[[[0, -1.],
                                                            [0, 1.]]]], device=self.device)

        self.laplacoan_filter.weight.data = torch.tensor([[[[0, -1., 0],
                                                            [-1., 4., -1.],
                                                            [0, -1., 0]]]],  device=self.device)

    def forward(self, gen_output, p_next_step):
        # vel_grad = self.get_velocity_grad(gen_output)
        # vel_grad_HR = self.get_velocity_grad(HR_image)
        # strain_rate_2_HR = torch.mean(torch.mean(self.get_strain_rate_mags(
        # vel_grad_HR), dim=2, keepdim=True), dim=3, keepdim=True)

        # continuity loss
        continuity_res = self.get_continuity_res(gen_output)
        continuity_loss = torch.mean(
            torch.abs(continuity_res), dim=(2, 3))

        # poisson loss
        dudt = self.get_poisson_dudt(gen_output)
        dvdt = self.get_poisson_dvdt(gen_output)

        # stationary loss
        # poisson_loss = torch.mean(torch.abs(dudt), dim=(2, 3)) + \
        #     torch.mean(torch.abs(dvdt), dim=(2, 3))

        # unstationary loss
        p_prime = p_next_step - gen_output[:, 2, :, :].unsqueeze(1)
        dudt_pad = F.pad(dudt, (1, 2, 1, 1), "constant", 0.0)
        dvdt_pad = F.pad(dvdt, (1, 1, 1, 2), "constant", 0.0)

        u_new_for = torch.zeros_like(p_next_step)
        v_new_for = torch.zeros_like(p_next_step)

        for i in range(1, self.N):
            for j in range(1, self.N + 1):
                u_new_for[:, :, j, i] = gen_output[:, 0, j, i].unsqueeze(1) + \
                    dudt_pad[:, :, j, i] * self.dt
        for i in range(1, self.N + 1):
            for j in range(1, self.N):
                v_new_for[:, :, j, i] = gen_output[:, 1, j, i].unsqueeze(1) + \
                    dvdt_pad[:, :, j, i] * self.dt

        b_for = torch.zeros_like(p_next_step)
        poisson_loss_for = torch.zeros_like(p_next_step)

        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                b_for[:, :, j, i] = (1. / self.dt) * ((u_new_for[:, :, j, i] - u_new_for[:, :, j, i - 1]) / self.dx +
                                                      (v_new_for[:, :, j, i] - v_new_for[:, :, j - 1, i]) / self.dx)

        for i in range(1, self.N + 1):
            for j in range(1, self.N + 1):
                a_p = 4./(self.dx ** 2)
                a_e = 1./(self.dx ** 2)
                a_s = 1./(self.dx ** 2)
                a_w = 1./(self.dx ** 2)
                a_n = 1./(self.dx ** 2)
                if i == 1:
                    a_p = a_p - a_w
                    a_w = 0
                if i == self.N:
                    a_p = a_p - a_e
                    a_e = 0
                if j == 1:
                    a_p = a_p - a_s
                    a_s = 0
                if j == self.N:
                    a_p = a_p - a_n
                    a_n = 0
                poisson_loss_for[:, :, j, i] = a_p * p_prime[:, :, j, i] - \
                    a_e * p_prime[:, :, j, i+1] - \
                    a_s * p_prime[:, :, j-1, i] - \
                    a_w * p_prime[:, :, j, i-1] - \
                    a_n * p_prime[:, :, j+1, i] + b_for[:, :, j, i]

        for_max = torch.max(poisson_loss_for[:, :, 1:-1, 1:-1])
        for_max_no_bc = torch.max(poisson_loss_for)

        u_new_conv = gen_output[:, 0, :, :].unsqueeze(1) + dudt_pad * self.dt
        v_new_conv = gen_output[:, 1, :, :].unsqueeze(1) + dvdt_pad * self.dt

        b_conv = (1. / self.dt) * (self.div_ddx(u_new_conv) /
                                   self.dx + self.div_ddy(v_new_conv) / self.dx)

        poisson_loss_conv = self.laplacoan_filter(
            p_prime) / (self.dx ** 2) + b_conv

        conv_max = torch.max(poisson_loss_conv[:, :, 1:-1, 1:-1])
        conv_max_no_bc = torch.max(poisson_loss_conv)

        # boundary loss
        b_c_loss = self.get_b_c_loss(gen_output)

        # pi_loss
        pi_loss = self.labmda_con * continuity_loss + \
            self.lambda_b_c * b_c_loss + \
            (1 - self.labmda_con - self.lambda_b_c) * poisson_loss
        pi_loss = torch.mean(pi_loss.view(-1))
        return pi_loss

    # def get_velocity_grad(self, input):
    #     dudx = self.ddx(input, 0)
    #     dvdx = self.ddx(input, 1)
    #     dudy = self.ddy(input, 0)
    #     dvdy = self.ddy(input, 1)
    #     return dudx, dvdx, dudy, dvdy

    # def get_strain_rate_mags(self, vel_grad):
    #     dudx, dvdx, dudy, dvdy = vel_grad

    #     strain_rate_mag2 = dudx**2 + dvdy**2 + 2 * \
    #         ((0.5 * (dudy + dvdx))**2)

    #     return strain_rate_mag2

    def get_continuity_res(self, gen_output):
        dudx = self.div_ddx(
            gen_output[:, 0, :, :].unsqueeze(1)) / self.dx
        dvdy = self.div_ddy(
            gen_output[:, 1, :, :].unsqueeze(1)) / self.dx
        return dudx + dvdy

    def get_poisson_dudt(self, gen_output):
        u = gen_output[:, 0, :, :].unsqueeze(1)
        v = gen_output[:, 1, :, :].unsqueeze(1)
        p = gen_output[:, 2, :, :].unsqueeze(1)

        ue = self.average_x(u)
        vn = self.average_x(v)

        flux_e = ue * \
            self.average_x(u) - (self.visc / self.dx) * self.poisson_up_ddx(u)
        flux_n = vn * \
            self.average_y(u) - (self.visc / self.dx) * self.poisson_up_ddy(u)

        flux_e = flux_e[:, :, :, :-1]
        flux_n = flux_n[:, :, :, :-1]
        p = p[:, :, :-1, 1:-1]

        flux_e_diff = -self.poisson_down_ddx(flux_e) / self.dx
        flux_n_diff = -self.poisson_down_ddy(flux_n) / self.dx
        p_diff = -self.poisson_down_ddx(p) / self.dx

        dudt = flux_e_diff + flux_n_diff + p_diff

        return dudt

    def get_poisson_dvdt(self, gen_output):
        u = gen_output[:, 0, :, :].unsqueeze(1)
        v = gen_output[:, 1, :, :].unsqueeze(1)
        p = gen_output[:, 2, :, :].unsqueeze(1)

        ue = self.average_y(u)
        vn = self.average_y(v)

        flux_e = ue * \
            self.average_x(v) - (self.visc / self.dx) * self.poisson_up_ddx(v)
        flux_n = vn * \
            self.average_y(v) - (self.visc / self.dx) * self.poisson_up_ddy(v)

        flux_e = flux_e[:, :, :-1, :]
        flux_n = flux_n[:, :, :-1, :]
        p = p[:, :, 1:-1, :-1]

        flux_e_diff = -self.poisson_down_ddx(flux_e) / self.dx
        flux_n_diff = -self.poisson_down_ddy(flux_n) / self.dx
        p_diff = -self.poisson_down_ddy(p) / self.dx

        dvdt = flux_e_diff + flux_n_diff + p_diff

        return dvdt

    def get_b_c_loss(self, gen_output):
        # y = 0
        y_0_loss = torch.sum(gen_output[:, 0, 0, 1:self.N] + gen_output[:, 0, 1, 1:self.N], dim=1) + \
            torch.sum(gen_output[:, 1, 0, 1:self.N + 1] +
                      gen_output[:, 2, 0, 1:self.N + 1], dim=1)
        # y = L
        y_L_loss = torch.sum(2 * self.u_0 - (gen_output[:, 0, self.N, 1:self.N] + gen_output[:, 0, self.N + 1, 1:self.N]), dim=1) + \
            torch.sum(gen_output[:, 1, self.N + 1, 1:self.N + 1] +
                      gen_output[:, 2, self.N + 1, 1:self.N + 1], dim=1)
        # x = 0
        x_0_loss = torch.sum(gen_output[:, 1, 1:self.N, 0] + gen_output[:, 1, 1:self.N, 1], dim=1) + \
            torch.sum(gen_output[:, 0, 1:self.N + 1, 0] +
                      gen_output[:, 2, 1:self.N + 1, 0], dim=1)
        # x = L
        x_L_loss = torch.sum(gen_output[:, 1, 1:self.N, self.N + 1] + gen_output[:, 1, 1:self.N, self.N], dim=1) + \
            torch.sum(gen_output[:, 0, 1:self.N + 1, self.N + 1] +
                      gen_output[:, 2, 1:self.N + 1, self.N + 1], dim=1)

        b_c_loss = torch.sum(
            (torch.abs(y_0_loss) +
             torch.abs(y_L_loss) +
             torch.abs(x_0_loss) +
             torch.abs(x_L_loss)).view(-1))
        return b_c_loss


if __name__ == '__main__':
    from data_utils import make_dataset_from_pickle
    from torch.utils.data import DataLoader

    train_dataset, valid_dataset = make_dataset_from_pickle(
        'data/1221_16000step/first_half.pickle', 4, 'PI_loss_test_dir', 2000)
    dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=True).__iter__()
    INDEX = 5
    if INDEX != 0:
        for i in range(INDEX - 1):
            dataloader.__next__()

    lambda_params = (0.5, 0.01)

    PI_loss = PhysicsInformedLoss(*lambda_params, *train_dataset.get_params())

    lr, hr, lr_expanded, p_next_step = dataloader.__next__()

    loss = PI_loss(hr, p_next_step)
    print(loss)
