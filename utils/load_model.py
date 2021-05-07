import torch
import torch.nn as nn
import torch.nn.functional as F


def count_model_params(model, is_trainable):
    if is_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters())

    return num_params


class SqueezeExcitationModule(nn.Module):
    def __init__(self, c_in, channel_factor=2):
        super(SqueezeExcitationModule, self).__init__()

        self.c_in = c_in
        self.channel_factor = channel_factor

        self.squeeze = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv_excitation_downsample = self.conv_1x1_block(
            c_in=self.c_in, c_out=self.c_in // self.channel_factor
        )
        self.conv_excitation_upsample = self.conv_1x1_block(
            c_in=self.c_in // self.channel_factor, c_out=self.c_in
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        o = self.squeeze(x)
        o = self.conv_excitation_downsample(o)
        o = self.conv_excitation_upsample(o)
        o = self.sigmoid(o)

        x *= o

        return x

    def conv_1x1_block(self, c_in, c_out):
        seq = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in, out_channels=c_out, kernel_size=1, stride=1, padding=0
            ),
            nn.ReLU(),
        )

        return seq


class ConvBlockResSqex(nn.Module):
    def __init__(self, c_in, c_out):
        super(ConvBlockResSqex, self).__init__()

        self.c_in = c_in
        self.c_out = c_out

        self.conv_1 = self.conv_block(
            c_in=self.c_in, c_out=self.c_in, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = self.conv_block(
            c_in=self.c_in, c_out=self.c_out, kernel_size=3, stride=1, padding=1
        )
        self.sqex = SqueezeExcitationModule(c_in=self.c_in)

    def forward(self, x):
        x_out = self.conv_1(x)
        x_out = self.sqex(x_out)
        x_out += x
        x_out = self.conv_2(x_out)

        return x_out

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.GroupNorm(num_groups=c_out // 2, num_channels=c_out),
            nn.ReLU(),
        )

        return seq_block


# ================================================================================
# ================================================================================


#############################################################
# DeconvNet Encoder Small
#############################################################


class EncoderDeconvSmall(nn.Module):
    def __init__(self):
        super(EncoderDeconvSmall, self).__init__()

        self.conv_init = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_1 = ConvBlockResSqex(c_in=32, c_out=64)
        self.conv_block_2 = ConvBlockResSqex(c_in=64, c_out=128)
        self.conv_block_3 = ConvBlockResSqex(c_in=128, c_out=256)
        self.conv_block_4 = ConvBlockResSqex(c_in=256, c_out=512)
        self.conv_block_5 = ConvBlockResSqex(c_in=512, c_out=1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        # 1 x 512 x 512
        x = self.conv_init(x)

        x1 = self.conv_block_1(x)  # 64 x 512 x 512

        size_1 = x1.shape
        x1, indices_1 = self.maxpool(x1)  # 64 x 256 x 256

        x2 = self.conv_block_2(x1)  # 128 x 256 x 256

        size_2 = x2.shape
        x2, indices_2 = self.maxpool(x2)  # 128 x 128 x 128

        x3 = self.conv_block_3(x2)  # 256 x 128 x 128

        size_3 = x3.shape
        x3, indices_3 = self.maxpool(x3)  # 256 x 64 x 64

        x4 = self.conv_block_4(x3)  # 512 x 64 x 64

        size_4 = x4.shape
        x4, indices_4 = self.maxpool(x4)  # 512 x 32 x 32

        x5 = self.conv_block_5(x4)  # 1024 x 32 x 32

        size_5 = x5.shape
        x5, indices_5 = self.maxpool(x5)  # 1024 x 16 x 16

        pool_indices_dict = {
            "indices_1": indices_1,
            "indices_2": indices_2,
            "indices_3": indices_3,
            "indices_4": indices_4,
            "indices_5": indices_5,
        }

        pool_size_dict = {
            "size_1": size_1,
            "size_2": size_2,
            "size_3": size_3,
            "size_4": size_4,
            "size_5": size_5,
        }

        return x5, pool_indices_dict, pool_size_dict


#############################################################
# DeconvNet Decoder Small
#############################################################


class DecoderDeconvSmall(nn.Module):
    def __init__(self):
        super(DecoderDeconvSmall, self).__init__()

        self.deconv_block_5 = ConvBlockResSqex(c_in=1024, c_out=512)
        self.deconv_block_4 = ConvBlockResSqex(c_in=512, c_out=256)
        self.deconv_block_3 = ConvBlockResSqex(c_in=256, c_out=128)
        self.deconv_block_2 = ConvBlockResSqex(c_in=128, c_out=64)
        self.deconv_block_1 = ConvBlockResSqex(c_in=64, c_out=32)

        self.transpose_conv = nn.ConvTranspose2d(
            in_channels=1024, out_channels=1024, kernel_size=2, stride=2
        )

        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, pool_indices_dict, pool_size_dict):
        # 1024 x 16 x 16

        x = self.transpose_conv(x)  # 1024 x 32 x 32

        # x = self.max_unpool(
        #     x, pool_indices_dict["indices_5"], output_size=pool_size_dict["size_5"]
        # )  # 1024 x 32 x 32

        x = self.deconv_block_5(x)  # 512 x 32 x 32

        x = self.max_unpool(
            x, pool_indices_dict["indices_4"], output_size=pool_size_dict["size_4"]
        )  # 512 x 64 x 64

        x = self.deconv_block_4(x)  # 256 x 64 x 64

        x = self.max_unpool(
            x, pool_indices_dict["indices_3"], output_size=pool_size_dict["size_3"]
        )  # 256 x 128 x 128

        x = self.deconv_block_3(x)  # 128 x 128 x 128

        x = self.max_unpool(
            x, pool_indices_dict["indices_2"], output_size=pool_size_dict["size_2"]
        )  # 128 x 224 x 2242

        x = self.deconv_block_2(x)  # 64 x 224 x 224

        x = self.max_unpool(
            x, pool_indices_dict["indices_1"], output_size=pool_size_dict["size_1"]
        )  # 64 x 512 x 512

        o = self.deconv_block_1(x)  # 64 x 512 x 512

        return o


#############################################################
# DeconvNet Model Small
#############################################################


class ModelDeconvSmall(nn.Module):
    def __init__(self, num_class):
        super(ModelDeconvSmall, self).__init__()
        self.num_class = num_class

        self.encoder = EncoderDeconvSmall()
        self.decoder = DecoderDeconvSmall()

        self.final_conv = nn.Conv2d(
            in_channels=32,
            out_channels=self.num_class,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # 1 x 512 x 512
        x, pool_indices_dict, pool_size_dict = self.encoder(x)  # 1024 x 1 x 1
        x = self.decoder(x, pool_indices_dict, pool_size_dict)  # 64 x 512 x 512

        output = self.final_conv(x)  # num_class x 512 x 512

        return output


# ================================================================================
# ================================================================================


#############################################################
# DeconvNet Encoder
#############################################################


class DeconvEncoder(nn.Module):
    def __init__(self):
        super(DeconvEncoder, self).__init__()

        self.conv_block_1_1 = self.conv_block(
            c_in=1, c_out=64, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_1_2 = self.conv_block(
            c_in=64, c_out=64, kernel_size=3, stride=1, padding=1
        )

        self.conv_block_2_1 = self.conv_block(
            c_in=64, c_out=128, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_2_2 = self.conv_block(
            c_in=128, c_out=128, kernel_size=3, stride=1, padding=1
        )

        self.conv_block_3_1 = self.conv_block(
            c_in=128, c_out=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_3_2 = self.conv_block(
            c_in=256, c_out=256, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_3_3 = self.conv_block(
            c_in=256, c_out=256, kernel_size=3, stride=1, padding=1
        )

        self.conv_block_4_1 = self.conv_block(
            c_in=256, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_4_2 = self.conv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_4_3 = self.conv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )

        self.conv_block_5_1 = self.conv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_5_2 = self.conv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_5_3 = self.conv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )

        self.conv_fc6 = nn.Conv2d(
            in_channels=512, out_channels=1024, kernel_size=16, stride=1, padding=0
        )
        self.conv_fc7 = nn.Conv2d(
            in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        # 3 x 512 x 512
        x = self.conv_block_1_1(x)  # 64 x 512 x 512
        x = self.conv_block_1_2(x)  # 64 x 512 x 512

        size_1 = x.shape
        x, indices_1 = self.maxpool(x)  # 64 x 256 x 256

        x = self.conv_block_2_1(x)  # 128 x 256 x 256
        x = self.conv_block_2_2(x)  # 128 x 256 x 256

        size_2 = x.shape
        x, indices_2 = self.maxpool(x)  # 128 x 128 x 128

        x = self.conv_block_3_1(x)  # 256 x 128 x 128
        x = self.conv_block_3_2(x)  # 256 x 128 x 128
        x = self.conv_block_3_3(x)  # 256 x 128 x 128

        size_3 = x.shape
        x, indices_3 = self.maxpool(x)  # 256 x 64 x 64

        x = self.conv_block_4_1(x)  # 512 x 64 x 64
        x = self.conv_block_4_2(x)  # 512 x 64 x 64
        x = self.conv_block_4_3(x)  # 512 x 64 x 64

        size_4 = x.shape
        x, indices_4 = self.maxpool(x)  # 512 x 28 x 28

        x = self.conv_block_5_1(x)  # 512 x 28 x 28
        x = self.conv_block_5_2(x)  # 512 x 28 x 28
        x = self.conv_block_5_3(x)  # 512 x 28 x 28

        size_5 = x.shape
        x, indices_5 = self.maxpool(x)  # 512 x 14 x 14

        x = self.conv_fc6(x)  # 1024 x 1 x 1
        o = self.conv_fc7(x)  # 1024 x 1 x 1

        pool_indices_dict = {
            "indices_1": indices_1,
            "indices_2": indices_2,
            "indices_3": indices_3,
            "indices_4": indices_4,
            "indices_5": indices_5,
        }

        pool_size_dict = {
            "size_1": size_1,
            "size_2": size_2,
            "size_3": size_3,
            "size_4": size_4,
            "size_5": size_5,
        }

        return o, pool_indices_dict, pool_size_dict

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.GroupNorm(num_groups=c_out // 2, num_channels=c_out),
            nn.ReLU(),
        )

        return seq_block


#############################################################
# DeconvNet Decoder
#############################################################


class DeconvDecoder(nn.Module):
    def __init__(self):
        super(DeconvDecoder, self).__init__()

        self.deconv_fc6 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=14, stride=1, padding=0
        )

        self.deconv_block_5_3 = self.deconv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_5_2 = self.deconv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_5_1 = self.deconv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )

        self.deconv_block_4_3 = self.deconv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_4_2 = self.deconv_block(
            c_in=512, c_out=512, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_4_1 = self.deconv_block(
            c_in=512, c_out=256, kernel_size=3, stride=1, padding=1
        )

        self.deconv_block_3_3 = self.deconv_block(
            c_in=256, c_out=256, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_3_2 = self.deconv_block(
            c_in=256, c_out=256, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_3_1 = self.deconv_block(
            c_in=256, c_out=128, kernel_size=3, stride=1, padding=1
        )

        self.deconv_block_2_2 = self.deconv_block(
            c_in=128, c_out=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_2_1 = self.deconv_block(
            c_in=128, c_out=64, kernel_size=3, stride=1, padding=1
        )

        self.deconv_block_1_2 = self.deconv_block(
            c_in=64, c_out=64, kernel_size=3, stride=1, padding=1
        )
        self.deconv_block_1_1 = self.deconv_block(
            c_in=64, c_out=64, kernel_size=3, stride=1, padding=1
        )

        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x, pool_indices_dict, pool_size_dict):
        # 1024 x 1 x1
        x = self.deconv_fc6(x)  # 512 x 7 x 7

        x = self.max_unpool(
            x, pool_indices_dict["indices_5"], output_size=pool_size_dict["size_5"]
        )  # 512 x 14 x 14

        x = self.deconv_block_5_3(x)  # 512 x 14 x 14
        x = self.deconv_block_5_2(x)  # 512 x 14 x 14
        x = self.deconv_block_5_1(x)  # 512 x 14 x 14

        x = self.max_unpool(
            x, pool_indices_dict["indices_4"], output_size=pool_size_dict["size_4"]
        )  # 512 x 28 x 28

        x = self.deconv_block_4_3(x)  # 512 x 28 x 28
        x = self.deconv_block_4_2(x)  # 512 x 28 x 28
        x = self.deconv_block_4_1(x)  # 256 x 28 x 28

        x = self.max_unpool(
            x, pool_indices_dict["indices_3"], output_size=pool_size_dict["size_3"]
        )  # 256 x 56 x 56

        x = self.deconv_block_3_3(x)  # 256 x 56 x 56
        x = self.deconv_block_3_2(x)  # 256 x 56 x 56
        x = self.deconv_block_3_1(x)  # 128 x 56 x 56

        x = self.max_unpool(
            x, pool_indices_dict["indices_2"], output_size=pool_size_dict["size_2"]
        )  # 128 x 112 x 112

        x = self.deconv_block_2_2(x)  # 128 x 112 x 112
        x = self.deconv_block_2_1(x)  # 64 x 112 x 112

        x = self.max_unpool(
            x, pool_indices_dict["indices_1"], output_size=pool_size_dict["size_1"]
        )  # 64 x 224 x 224

        x = self.deconv_block_1_2(x)  # 64 x 224 x 224
        o = self.deconv_block_1_1(x)  # 64 x 224 x 224

        return o

    def deconv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.GroupNorm(num_groups=c_out // 2, num_channels=c_out),
            nn.ReLU(),
        )

        return seq_block


#############################################################
# DeconvNet Model
#############################################################


class ModelDeconv(nn.Module):
    def __init__(self, num_class):
        super(ModelDeconv, self).__init__()
        self.num_class = num_class

        self.encoder = DeconvEncoder()
        self.decoder = DeconvDecoder()

        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=self.num_class,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x):
        # 3 x 224 x 224
        x, pool_indices_dict, pool_size_dict = self.encoder(x)  # 1024 x 1 x 1
        x = self.decoder(x, pool_indices_dict, pool_size_dict)  # 64 x 224 x 224

        output = self.final_conv(x)  # num_class x 224 x 224

        return output


# ================================================================================
# ================================================================================

#############################################################
# FCN8 Encoder
#############################################################


class ModelFcn8Encoder(nn.Module):
    def __init__(self):
        super(ModelFcn8Encoder, self).__init__()

        self.block_1 = self.conv_block(
            c_in=1, c_out=32, kernel_size=3, stride=1, dilation=4, padding=4
        )

        self.block_2 = self.conv_block(
            c_in=32, c_out=64, kernel_size=3, stride=1, dilation=3, padding=3
        )

        self.block_3 = self.conv_block(
            c_in=64, c_out=128, kernel_size=3, stride=1, dilation=2, padding=2
        )

        self.block_4 = self.conv_block(
            c_in=128, c_out=256, kernel_size=3, stride=1, dilation=1, padding=1
        )

        self.block_5 = self.conv_block(
            c_in=256, c_out=512, kernel_size=3, stride=1, padding=1
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 1 x 512 x 512
        x1 = self.block_1(x)  # 32 x 512 x 512
        p1 = self.maxpool(x1)  # 32 x 256 x 256

        x2 = self.block_2(p1)  # 64 x 256 x256
        p2 = self.maxpool(x2)  # 64 x 128 x 128

        x3 = self.block_3(p2)  # 128 x 128 x 128
        p3 = self.maxpool(x3)  # 128 x 64 x 64

        x4 = self.block_4(p3)  # 256 x 64 x 64
        p4 = self.maxpool(x4)  # 256 x 32 x 32

        x5 = self.block_5(p4)  # 512 x 32 x 32
        p5 = self.maxpool(x5)  # 512 x 16 x 16

        return p5, p4, p3

    def conv_block(self, c_in, c_out, **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.GroupNorm(num_groups=c_out // 2, num_channels=c_out),
            # nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )

        return seq_block


#############################################################
# FCN8 Decoder
#############################################################
class ModelFcn8Decoder(nn.Module):
    def __init__(self):
        super(ModelFcn8Decoder, self).__init__()

        self.upsample_5 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2,
        )

        self.upsample_4 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2,
        )

        self.upsample_3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=8, stride=8,
        )

    def forward(self, x5, x4, x3):
        # x5 = 512 x 16 x 16
        o5 = self.upsample_5(x5)  # 256 x 32 x 32
        o4 = self.upsample_4(o5 + x4)  # 128 x 64 x 64
        o3 = self.upsample_3(o4 + x3)  # 64 x 512 x 512

        return o3


#############################################################
# FCN8 Model
#############################################################


class ModelFcn8(nn.Module):
    def __init__(self, num_class):
        super(ModelFcn8, self).__init__()

        self.num_class = num_class
        self.encoder = ModelFcn8Encoder()
        self.decoder = ModelFcn8Decoder()

        self.final_conv = nn.Conv2d(
            64, self.num_class, kernel_size=1, stride=1, padding=0
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x5, x4, x3 = self.encoder(x)
        o = self.decoder(x5, x4, x3)

        output = self.final_conv(o)

        return output


if __name__ == "__main__":
    print("Model sizes are:")

    model_list = [
        ModelDeconv(num_class=2),
        ModelFcn8(num_class=2),
        ModelDeconvSmall(num_class=2),
    ]

    max_pad_print = max([len(i.__class__.__name__) for i in model_list])

    print(f"{' ' * (max_pad_print+5)}Trainable {' ' * 2} NonTrainable")
    for m in model_list:
        print(
            f"{m.__class__.__name__ :{max_pad_print}} : {count_model_params(m, is_trainable=True):>12,d} | {count_model_params(m, is_trainable=False):>12,d}"
        )
