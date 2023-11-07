import torch
import torch.nn as nn

from torch.nn.functional import relu

ReLU = nn.ReLU(inplace=True)
from Ftool import half_chaifen, half_hecheng
###修改后的复数网络，是在内部进行合并差分的
###relu替换为了SiLU
class Complex_relu(nn.Module):
    def __init__(self):
        super(Complex_relu, self).__init__()

    def forward(self, x):
        real_x, imag_x, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        real_x = ReLU(real_x)
        imag_x = ReLU(imag_x)
        r = torch.real(half_hecheng(real_x, imag_x,pre_row_real, pre_col_real, pre_row_imag, pre_col_imag, x.shape[3]))
        return r


class Complex_Leakyrelu(nn.Module):
    def __init__(self):
        super(Complex_Leakyrelu, self).__init__()

    def forward(self, input_real, input_imag):
        return nn.LeakyReLU(input_real), nn.LeakyReLU(input_imag)

###修改后的复数网络，拆分后内部的pre_row_real, pre_col_real, pre_row_imag, pre_col_imag的第三和第四维度会缩小，还没想到解决方法
class Complex_maxpooling2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(Complex_maxpooling2d, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.conv_real = nn.Conv2d(in_channels, in_channels * 2, kernel_size, stride)
        self.conv_imag = nn.Conv2d(in_channels, in_channels * 2, kernel_size, stride)
    def forward(self, x):
        input_real, input_imag, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        x = torch.complex(input_real, input_imag)
        abs_x = torch.abs(x)
        angle_x = torch.angle(x)
        big_abs_x = abs_x * 10
        sum_x = big_abs_x + angle_x
        abs_x = self.maxpool(abs_x)
        sum_x = self.maxpool(sum_x)
        angle_x = sum_x - (abs_x * 10)
        real_x = torch.mul(abs_x, torch.cos(angle_x))
        imag_x = torch.mul(abs_x, torch.sin(angle_x))
        ####POOLing后要卷积上去，out通道要乘以2

        r = torch.real(half_hecheng(real_x, imag_x, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag, x.shape[3]))
        return r

###修改后的复数网络，是在内部进行合并差分的
# 复数卷积块：封装复数卷积块
class complex_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(complex_conv_block, self).__init__()

        self.conv1 = ComplexConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.batch_norm2d = ComplexBatchNorm2d(ch_out)
        self.relu = Complex_relu()
        self.conv2 = ComplexConv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        x_real, x_imag, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        x_real, x_imag = self.conv1(x_real, x_imag)
        x_real, x_imag = self.batch_norm2d(x_real, x_imag)
        x_real, x_imag = self.relu(x_real, x_imag)
        x_real, x_imag = self.conv2(x_real, x_imag)
        x_real, x_imag = self.batch_norm2d(x_real, x_imag)
        x_real, x_imag = self.relu(x_real, x_imag)
        r = torch.real(half_hecheng(x_real, x_imag, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag, x.shape[3]))
        return r

###修改后的复数网络，是在内部进行合并差分的
# 复数卷积层
# 一个复数被拆分为两个实值进行操作，其中conv_real处理复数的实部，conv_imag处理复数的虚部
# 输入为real Feature map和imaginary Feature map，conv_real相当于论文中Kr，conv_imag相当于论文中Ki
# 网络输入分为实、虚，输出也为实、虚，通过复数的运算法则锚定了相位关系
class ComplexConv2d_init(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ComplexConv2d_init, self).__init__()
        self.conv_real = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.conv_imag = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.output_channels = output_channels

    def forward(self, x):
        input_real, input_imag, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        assert input_real.shape == input_imag.shape

        size_r = int((self.conv_real(input_real).shape[1]) / (pre_row_imag.shape[1]))
        r = torch.real(half_hecheng((self.conv_real(input_real) - self.conv_imag(input_imag)),
                                    (self.conv_imag(input_real) + self.conv_real(input_imag)),
                                    (pre_row_real).repeat(1, size_r, 1, 1),
                                    (pre_col_real).repeat(1, size_r, 1, 1),
                                    (pre_row_imag).repeat(1, size_r, 1, 1),
                                    (pre_col_imag).repeat(1, size_r, 1, 1), x.shape[3]))
        return r

class ComplexConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_real = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.conv_imag = nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, groups,
                                   bias)
        self.output_channels = output_channels

    def forward(self, x):
        input_real, input_imag, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        assert input_real.shape == input_imag.shape
        #r = torch.real(half_hecheng((self.conv_real(input_real) - self.conv_imag(input_imag)), (self.conv_imag(input_real) + self.conv_real(input_imag)), (pre_row_real), (pre_col_real), (pre_row_imag), (pre_col_imag), x.shape[3]))
        ##上采样的时候通道倍增，下采样的时候通道缩小
        if int(self.conv_real(input_real).shape[1]) > int(pre_row_imag.shape[1]):
            size_r = int((self.conv_real(input_real).shape[1]) / (pre_row_imag.shape[1]))
            r = torch.real(half_hecheng((self.conv_real(input_real) - self.conv_imag(input_imag)),
                                        (self.conv_imag(input_real) + self.conv_real(input_imag)),
                                        (pre_row_real).repeat(1, size_r, 1, 1),
                                        (pre_col_real).repeat(1, size_r, 1, 1),
                                        (pre_row_imag).repeat(1, size_r, 1, 1),
                                        (pre_col_imag).repeat(1, size_r, 1, 1), x.shape[3]))
            # for h in range (128):
            #     #注意一下h:(h+1+int(size_r/128)是取的几个数
            #     r[:,h,:,:] = torch.real(half_hecheng((self.conv_real(input_real) - self.conv_imag(input_imag))[:,h:(h+1+int(size_r/128)),:,:],
            #                                          (self.conv_imag(input_real) + self.conv_real(input_imag))[:,h:(h+1+int(size_r/128)),:,:],
            #                                          (pre_row_real).repeat(1, int(size_r/128), 1, 1),
            #                                          (pre_col_real).repeat(1, int(size_r/128), 1, 1),
            #                                          (pre_row_imag).repeat(1, int(size_r/128), 1, 1),
            #                                          (pre_col_imag).repeat(1, int(size_r/128), 1, 1), x.shape[3]))
        else:
            size_r = int(self.conv_real(input_real).shape[1])
            r = torch.real(half_hecheng((self.conv_real(input_real) - self.conv_imag(input_imag)),
                                        (self.conv_imag(input_real) + self.conv_real(input_imag)),
                                        torch.narrow((pre_row_real), 1, 0, size_r),
                                        torch.narrow((pre_col_real), 1, 0, size_r),
                                        torch.narrow((pre_row_imag), 1, 0, size_r),
                                        torch.narrow((pre_col_imag), 1, 0, size_r), x.shape[3]))
        return r


# 复数归一化层
class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, 2))
            self.register_buffer('running_covar', torch.zeros(num_features, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, x):
        input_r, input_i, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag = half_chaifen(x, x.shape[3])
        assert (input_r.size() == input_i.size())
        assert (len(input_r.shape) == 4)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])

            mean = torch.stack((mean_r, mean_i), dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r - mean_r[None, :, None, None]
            input_i = input_i - mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1. / n * input_r.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input_i.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0, 2, 3])

            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

            input_r = input_r - mean[None, :, 0, None, None]
            input_i = input_i - mean[None, :, 1, None, None]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_r, input_i = Rrr[None, :, None, None] * input_r + Rri[None, :, None, None] * input_i, \
                           Rii[None, :, None, None] * input_i + Rri[None, :, None, None] * input_r

        if self.affine:
            input_r, input_i = self.weight[None, :, 0, None, None] * input_r + self.weight[None, :, 2, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 0, None, None], \
                               self.weight[None, :, 2, None, None] * input_r + self.weight[None, :, 1, None,
                                                                               None] * input_i + \
                               self.bias[None, :, 1, None, None]
        r = torch.real(half_hecheng(input_r, input_i, pre_row_real, pre_col_real, pre_row_imag, pre_col_imag, x.shape[3]))
        return r


# 复数上采样层
class Complex_up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Complex_up_conv, self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.complex_cov = ComplexConv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.BN2d = ComplexBatchNorm2d(ch_out)
        self.relu = Complex_relu()

    def forward(self, real_x, imag_x):
        real_x = self.up(real_x)
        imag_x = self.up(imag_x)
        real_x, imag_x = self.complex_cov(real_x, imag_x)
        real_x, imag_x = self.BN2d(real_x, imag_x)
        real_x, imag_x = self.relu(real_x, imag_x)
        return real_x, imag_x

# 64 Channel spectrum feature extraction
class complex_unet2d(nn.Module):
    def __init__(self):
        super(complex_unet2d, self).__init__()
        numberchannel = 32
        self.Maxpool = Complex_maxpooling2d(kernel_size=2, stride=2)
        self.Conv1 = complex_conv_block(ch_in=1 * 2, ch_out=numberchannel)
        self.Conv2 = complex_conv_block(ch_in=numberchannel, ch_out=2 * numberchannel)
        self.Conv3 = complex_conv_block(ch_in=2 * numberchannel, ch_out=4 * numberchannel)
        self.Conv4 = complex_conv_block(ch_in=4 * numberchannel, ch_out=8 * numberchannel)
        self.Conv5 = complex_conv_block(ch_in=8 * numberchannel, ch_out=16 * numberchannel)

        self.Up5 = Complex_up_conv(ch_in=16 * numberchannel, ch_out=8 * numberchannel)
        self.Up_conv5 = complex_conv_block(ch_in=16 * numberchannel, ch_out=8 * numberchannel)

        self.Up4 = Complex_up_conv(ch_in=8 * numberchannel, ch_out=4 * numberchannel)
        self.Up_conv4 = complex_conv_block(ch_in=8 * numberchannel, ch_out=4 * numberchannel)

        self.Up3 = Complex_up_conv(ch_in=4 * numberchannel, ch_out=2 * numberchannel)
        self.Up_conv3 = complex_conv_block(ch_in=4 * numberchannel, ch_out=2 * numberchannel)

        self.Up2 = Complex_up_conv(ch_in=2 * numberchannel, ch_out=numberchannel)
        self.Up_conv2 = complex_conv_block(ch_in=2 * numberchannel, ch_out=numberchannel)

        self.Conv_1x1 = ComplexConv2d(numberchannel, 1 * 2, kernel_size=1, stride=1, padding=0)

    def forward(self, real_x, imag_x):
        real_x1, imag_x1 = self.Conv1(real_x, imag_x)

        real_x2, imag_x2 = self.Maxpool(real_x1, imag_x1)
        real_x2, imag_x2 = self.Conv2(real_x2, imag_x2)

        real_x3, imag_x3 = self.Maxpool(real_x2, imag_x2)
        real_x3, imag_x3 = self.Conv3(real_x3, imag_x3)

        real_x4, imag_x4 = self.Maxpool(real_x3, imag_x3)
        real_x4, imag_x4 = self.Conv4(real_x4, imag_x4)

        real_x5, imag_x5 = self.Maxpool(real_x4, imag_x4)
        real_x5, imag_x5 = self.Conv5(real_x5, imag_x5)

        real_d5, imag_d5 = self.Up5(real_x5, imag_x5)
        real_d5 = torch.cat((real_x4, real_d5), dim=1)
        imag_d5 = torch.cat((imag_x4, imag_d5), dim=1)
        real_d5, imag_d5 = self.Up_conv5(real_d5, imag_d5)

        real_d4, imag_d4 = self.Up4(real_d5, imag_d5)
        real_d4 = torch.cat((real_x3, real_d4), dim=1)
        imag_d4 = torch.cat((imag_x3, imag_d4), dim=1)
        real_d4, imag_d4 = self.Up_conv4(real_d4, imag_d4)

        real_d3, imag_d3 = self.Up3(real_d4, imag_d4)
        real_d3 = torch.cat((real_x2, real_d3), dim=1)
        imag_d3 = torch.cat((imag_x2, imag_d3), dim=1)
        real_d3, imag_d3 = self.Up_conv3(real_d3, imag_d3)

        real_d2, imag_d2 = self.Up2(real_d3, imag_d3)
        real_d2 = torch.cat((real_x1, real_d2), dim=1)
        imag_d2 = torch.cat((imag_x1, imag_d2), dim=1)
        real_d2, imag_d2 = self.Up_conv2(real_d2, imag_d2)

        real_d1, imag_d1 = self.Conv_1x1(real_d2, imag_d2)
        real_output = real_d1 + real_x
        imag_output = imag_d1 + imag_x

        return real_output, imag_output

# 32 Channel spectrum feature extraction
class complex_unet2d1(nn.Module):
    def __init__(self):
        super(complex_unet2d1, self).__init__()
        numberchannel = 32
        self.Maxpool = Complex_maxpooling2d(kernel_size=2, stride=2)
        self.Conv1 = complex_conv_block(ch_in=1, ch_out=numberchannel)
        self.Conv2 = complex_conv_block(ch_in=numberchannel, ch_out=2 * numberchannel)
        self.Conv3 = complex_conv_block(ch_in=2 * numberchannel, ch_out=4 * numberchannel)
        self.Conv4 = complex_conv_block(ch_in=4 * numberchannel, ch_out=8 * numberchannel)
        self.Conv5 = complex_conv_block(ch_in=8 * numberchannel, ch_out=16 * numberchannel)

        self.Up5 = Complex_up_conv(ch_in=16 * numberchannel, ch_out=8 * numberchannel)
        self.Up_conv5 = complex_conv_block(ch_in=16 * numberchannel, ch_out=8 * numberchannel)

        self.Up4 = Complex_up_conv(ch_in=8 * numberchannel, ch_out=4 * numberchannel)
        self.Up_conv4 = complex_conv_block(ch_in=8 * numberchannel, ch_out=4 * numberchannel)

        self.Up3 = Complex_up_conv(ch_in=4 * numberchannel, ch_out=2 * numberchannel)
        self.Up_conv3 = complex_conv_block(ch_in=4 * numberchannel, ch_out=2 * numberchannel)

        self.Up2 = Complex_up_conv(ch_in=2 * numberchannel, ch_out=numberchannel)
        self.Up_conv2 = complex_conv_block(ch_in=2 * numberchannel, ch_out=numberchannel)

        self.Conv_1x1 = ComplexConv2d(numberchannel, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, real_x, imag_x):
        real_x1, imag_x1 = self.Conv1(real_x, imag_x)

        real_x2, imag_x2 = self.Maxpool(real_x1, imag_x1)
        real_x2, imag_x2 = self.Conv2(real_x2, imag_x2)

        real_x3, imag_x3 = self.Maxpool(real_x2, imag_x2)
        real_x3, imag_x3 = self.Conv3(real_x3, imag_x3)

        real_x4, imag_x4 = self.Maxpool(real_x3, imag_x3)
        real_x4, imag_x4 = self.Conv4(real_x4, imag_x4)

        real_x5, imag_x5 = self.Maxpool(real_x4, imag_x4)
        real_x5, imag_x5 = self.Conv5(real_x5, imag_x5)

        real_d5, imag_d5 = self.Up5(real_x5, imag_x5)
        real_d5 = torch.cat((real_x4, real_d5), dim=1)
        imag_d5 = torch.cat((imag_x4, imag_d5), dim=1)
        real_d5, imag_d5 = self.Up_conv5(real_d5, imag_d5)

        real_d4, imag_d4 = self.Up4(real_d5, imag_d5)
        real_d4 = torch.cat((real_x3, real_d4), dim=1)
        imag_d4 = torch.cat((imag_x3, imag_d4), dim=1)
        real_d4, imag_d4 = self.Up_conv4(real_d4, imag_d4)

        real_d3, imag_d3 = self.Up3(real_d4, imag_d4)
        real_d3 = torch.cat((real_x2, real_d3), dim=1)
        imag_d3 = torch.cat((imag_x2, imag_d3), dim=1)
        real_d3, imag_d3 = self.Up_conv3(real_d3, imag_d3)

        real_d2, imag_d2 = self.Up2(real_d3, imag_d3)
        real_d2 = torch.cat((real_x1, real_d2), dim=1)
        imag_d2 = torch.cat((imag_x1, imag_d2), dim=1)
        real_d2, imag_d2 = self.Up_conv2(real_d2, imag_d2)

        real_d1, imag_d1 = self.Conv_1x1(real_d2, imag_d2)
        real_output = real_d1 + real_x
        imag_output = imag_d1 + imag_x

        return real_output, imag_output
