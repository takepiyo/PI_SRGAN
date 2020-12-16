import torch
import torch.nn as nn


class PhysicsInformedLoss(nn.Module):

    def __init__(self, labmda_con, dx, dt):
        super(PhysicsInformedLoss, self).__init__()
        self.labmda_con = labmda_con
        self.dx = dx
        self.dt = dt
        self.first_order_diff_x = self.get_diff_filter(0, 1)
        self.first_order_diff_y = self.get_diff_filter(1, 1)
        self.second_order_diff_x = self.get_diff_filter(0, 2)
        self.second_order_diff_y = self.get_diff_filter(1, 2)

    def forward(self, gen_output, HR_image):
        vel_grad = self.get_velocity_grad(gen_output)
        vel_grad_HR = self.get_velocity_grad(HR_image)
        strain_rate_2_HR = torch.mean(torch.mean(self.get_strain_rate_mags(
            vel_grad_HR), dim=2, keepdim=True), dim=3, keepdim=True)

        # continuity_loss = self.get_continuity_loss(vel_grad)
        # pressure_loss = self.get_pressure_loss(gen_output, vel_grad)
        continuity_res = self.get_continuity_res(vel_grad)
        pressure_res = self.get_pressure_res(gen_output, vel_grad)
        # continuity_loss = torch.square(continuity_loss) / strain_rate_2_HR
        # pressure_loss = torch.square(pressure_loss) / strain_rate_2_HR ** 2
        continuity_loss = torch.sum(
            (continuity_res ** 2) / strain_rate_2_HR, dim=(2, 3))
        pressure_loss = torch.sum(
            (pressure_res ** 2) / (strain_rate_2_HR ** 2), dim=(2, 3))

        return self.labmda_con * continuity_loss + (1 - self.labmda_con) * pressure_loss

    def get_velocity_grad(self, input):
        dudx = self.ddx(input, 0)
        dvdx = self.ddx(input, 1)
        dudy = self.ddy(input, 0)
        dvdy = self.ddy(input, 1)
        return dudx, dvdx, dudy, dvdy

    def get_strain_rate_mags(self, vel_grad):
        dudx, dvdx, dudy, dvdy = vel_grad

        strain_rate_mag2 = dudx**2 + dvdy**2 + 2 * \
            ((0.5 * (dudy + dvdx))**2)

        return strain_rate_mag2

    # def get_continuity_loss(self, vel_grad):
    #     dudx, dvdx, dudy, dvdy = vel_grad
    #     return dudx + dvdy

    # def get_pressure_loss(self, input, vel_grad):
    #     dudx, dvdx, dudy, dvdy = vel_grad
    #     d2pdx2 = self.d2dx2(input, 2)
    #     d2pdy2 = self.d2dy2(input, 2)

    #     return d2pdx2 + d2pdy2 + dudx * dudx + dvdy * dvdy + 2 * (dudy * dvdx)

    def ddx(self, input, channel):
        return self.first_order_diff_x(input[:, channel, :, :].unsqueeze(1)) / self.dx

    def ddy(self, input, channel):
        return self.first_order_diff_y(input[:, channel, :, :].unsqueeze(1)) / self.dx

    def d2dx2(self, input, channel):
        return self.second_order_diff_x(input[:, channel, :, :].unsqueeze(1)) / self.dx ** 2

    def d2dy2(self, input, channel):
        return self.second_order_diff_y(input[:, channel, :, :].unsqueeze(1)) / self.dx ** 2

    def get_continuity_res(self, vel_grad):
        dudx, dvdx, dudy, dvdy = vel_grad
        return dudx + dvdy

    def get_pressure_res(self, gen_output, vel_grad):
        dudx, dvdx, dudy, dvdy = vel_grad
        d2pdx2 = self.d2dx2(gen_output, 2)
        d2pdy2 = self.d2dy2(gen_output, 2)

        res = d2pdx2 * d2pdx2 + d2pdy2 * d2pdy2
        res += dudx * dudx + dvdy * dvdy + 2 * dudx * dvdy
        res -= (dudx + dvdy) / self.dt

        return res

    def get_diff_filter(self, direction, order):
        '''
        direction = 0 : x
        direction = 1 : y
        '''
        filter = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        if order == 1:
            w_0 = -1.0
            w_1 = 1.0
            if direction == 0:
                weight = torch.tensor([[[[0.0, 0.0, 0.0],
                                         [0.0, w_0, w_1],
                                         [0.0, 0.0, 0.0]]]], dtype=torch.float32)
                filter.weight.data = weight
            elif direction == 1:
                weight = torch.tensor([[[[0.0, w_1, 0.0],
                                         [0.0, w_0, 0.0],
                                         [0.0, 0.0, 0.0]]]], dtype=torch.float32)
                filter.weight.data = weight
            else:
                print("direction is accepted 1 or 2")
                exit()

        elif order == 2:
            w_0 = -2.0
            w_1 = 1.0
            if direction == 0:
                weight = torch.tensor([[[[0.0, 0.0, 0.0],
                                         [w_1, w_0, w_1],
                                         [0.0, 0.0, 0.0]]]], dtype=torch.float32)
                filter.weight.data = weight
            elif direction == 1:
                weight = torch.tensor([[[[0.0, w_1, 0.0],
                                         [0.0, w_0, 0.0],
                                         [0.0, w_1, 0.0]]]], dtype=torch.float32)
                filter.weight.data = weight
            else:
                print("direction is accepted 1 or 2")
                exit()
        else:
            print("order is accepted 1 or 2")
            exit()

        return filter

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
