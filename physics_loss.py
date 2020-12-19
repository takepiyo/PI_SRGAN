import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):

    def __init__(self, labmda_con, dx, dt, u_0, crop_size, visc):
        super(PhysicsInformedLoss, self).__init__()
        self.labmda_con = labmda_con
        self.dx = dx
        self.dt = dt
        self.u_0 = u_0
        self.crop_size = crop_size
        self.visc = visc
        self.continuity_ddx = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.continuity_ddy = nn.Conv2d(1, 1, 3, 1, 0, bias=False)
        self.average_x = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.average_y = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.poission_up_ddx = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.poission_up_ddy = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.poission_down_ddx = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.poission_down_ddy = nn.Conv2d(1, 1, 2, 1, 0, bias=False)
        self.setting_weights()

    def setting_weights(self):
        self.continuity_ddx.weight.data = torch.tensor([[[[0, 0, 0],
                                                          [-1., 1., 0],
                                                          [0, 0, 0]]]])

        self.continuity_ddy.weight.data = torch.tensor([[[[0, -1., 0],
                                                          [0, 1, 0],
                                                          [0, 0, 0]]]])

        self.average_x.weight.data = torch.tensor([[[[0.5, 0.5],
                                                     [0., 0.]]]])

        self.average_y.weight.data = torch.tensor([[[[0.5, 0.],
                                                     [0.5, 0.]]]])

        self.poission_up_ddx.weight.data = torch.tensor([[[[-1., 1.],
                                                           [0, 0]]]])

        self.poission_up_ddy.weight.data = torch.tensor([[[[-1., 0],
                                                           [1., 0]]]])

        self.poission_down_ddx.weight.data = torch.tensor([[[[0., 0],
                                                             [-1., 1.]]]])

        self.poission_down_ddy.weight.data = torch.tensor([[[[0, -1.],
                                                             [0, 1.]]]])

    def forward(self, gen_output, HR_image):
        # vel_grad = self.get_velocity_grad(gen_output)
        # vel_grad_HR = self.get_velocity_grad(HR_image)
        # strain_rate_2_HR = torch.mean(torch.mean(self.get_strain_rate_mags(
        # vel_grad_HR), dim=2, keepdim=True), dim=3, keepdim=True)

        # continuity loss
        continuity_res = self.get_continuity_res(gen_output)
        continuity_loss = torch.mean(
            torch.sqrt(continuity_res ** 2), dim=(2, 3))

        # poisson loss
        dudt = self.get_poission_dudt(gen_output)
        dvdt = self.get_poission_dvdt(gen_output)
        poisson_loss = torch.mean(torch.sqrt(dudt**2), dim=(2, 3)) + \
            torch.mean(torch.sqrt(dvdt**2), dim=(2, 3))

        return self.labmda_con * continuity_loss + (1 - self.labmda_con) * poisson_loss

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
        dudx = self.continuity_ddx(gen_output[:, 0, :, :]) / self.dx
        dvdy = self.continuity_ddy(gen_output[:, 1, :, :]) / self.dx
        return dudx + dvdy

    def get_poission_dudt(self, gen_output):
        u = gen_output[:, 0, :, :]
        v = gen_output[:, 1, :, :]
        p = gen_output[:, 2, :, :]

        ue = self.average_x(u)
        vn = self.average_x(v)

        flux_e = ue * \
            self.average_x(u) - (self.visc / self.dx) * self.poission_up_ddx(u)
        flux_n = vn * \
            self.average_y(u) - (self.visc / self.dx) * self.poission_up_ddy(u)

        flux_e = flux_e[:, :, :, :-1]
        flux_n = flux_n[:, :, :, :-1]
        p = p[:, :, :-1, 1:-1]

        flux_e_diff = -self.poission_down_ddx(flux_e) / self.dx
        flux_n_diff = -self.poission_down_ddy(flux_n) / self.dx
        p_diff = -self.poission_down_ddx(p) / self.dx

        dudt = flux_e_diff + flux_n_diff + p_diff

        return dudt

    def get_poission_dvdt(self, gen_output):
        u = gen_output[:, 0, :, :]
        v = gen_output[:, 1, :, :]
        p = gen_output[:, 2, :, :]

        ue = self.average_y(u)
        vn = self.average_y(v)

        flux_e = ue * \
            self.average_x(v) - (self.visc / self.dx) * self.poission_up_ddx(v)
        flux_n = vn * \
            self.average_y(v) - (self.visc / self.dx) * self.poission_up_ddy(v)

        flux_e = flux_e[:, :, :-1, :]
        flux_n = flux_n[:, :, :-1, :]
        p = p[:, :, 1:-1, :-1]

        flux_e_diff = -self.poission_down_ddx(flux_e) / self.dx
        flux_n_diff = -self.poission_down_ddy(flux_n) / self.dx
        p_diff = -self.poission_down_ddy(p) / self.dx

        dvdt = flux_e_diff + flux_n_diff + p_diff

        return dvdt


if __name__ == '__main__':
    from data_utils import make_dataset_from_pickle
    from torch.utils.data import DataLoader

    train_dataset, valid_dataset = make_dataset_from_pickle(
        'data/1214_data.pickle', 4, 'PI_loss_test_dir', 1000)
    dataloader = DataLoader(train_dataset, batch_size=1,
                            shuffle=False).__iter__()
    INDEX = 5
    if INDEX != 0:
        for i in range(INDEX - 1):
            dataloader.__next__()

    PI_loss = PhysicsInformedLoss(1.0, train_dataset.dx, train_dataset.dt)

    lr, hr_restore, hr, lr_expanded = dataloader.__next__()

    loss = PI_loss(hr, hr)
    print(loss)

    # def get_diff_filter(self, direction, order):
    #     '''
    #     direction = 0 : x
    #     direction = 1 : y
    #     '''
    #     filter = nn.Conv2d(1, 1, 7, 1, 3, bias=False)
    #     if order == 1:
    #         w_0 = 0.
    #         w_1 = -3. / 4.
    #         w_2 = 3. / 20.
    #         w_3 = -1. / 60.
    #     elif order == 2:
    #         w_0 = -49. / 18.
    #         w_1 = 3. / 2.
    #         w_2 = -3. / 20.
    #         w_3 = 1. / 90.
    #     else:
    #         print("order is accepted 1 or 2")
    #         exit()
    #     if direction == 0:
    #         weight = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                [w_3, w_2, w_1, w_0, w_1, w_2, w_3],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    #         filter.weight = weight
    #     elif direction == 1:
    #         weight = torch.tensor([[0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_0, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_1, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_2, 0.0, 0.0, 0.0],
    #                                [0.0, 0.0, 0.0, w_3, 0.0, 0.0, 0.0]], dtype=torch.float32)
    #         filter.weight = weight
    #     else:
    #         print("direction is accepted 1 or 2")
    #         exit()

    #     return filter
