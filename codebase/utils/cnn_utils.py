import torch.nn as nn


def get_conv(conv_f,
             in_chan,
             out_chan,
             kernel=4,
             stride=2,
             padding=0,
             norm=False,
             relu=False):
    """
        Helper function which builds a conv_f -> norm -> relu module and skips
        the bias in conv_f when needed
        - Note that conv_f can be conv2d or convtranspose2d
    """
    layer = [conv_f(in_chan, out_chan, kernel, stride, padding, bias=not norm)]
    if norm:
        layer.append(nn.BatchNorm2d(out_chan))
    if relu:
        layer.append(nn.ReLU())
    return layer


def get_layer(conv_f, channels, ker_sizes, strides, paddings, norms, relus):
    """
        Takes in channels, ker_sizes, strides and paddings as lists 
        where len(ker_sizes) == len(strides) == len(paddings) == len(channels) - 1
        returns of nn modules to splat into nn.Sequential
    """
    layer = []
    for i in range(len(channels) - 1):
        in_chan, out_chan = channels[i], channels[i + 1]
        kernel_size, stride, padding, norm, relu = ker_sizes[i], strides[i], paddings[i], \
                                                    norms[i], relus[i]
        layer += get_conv(
            conv_f,
            in_chan,
            out_chan,
            kernel=kernel_size,
            stride=stride,
            padding=padding,
            norm=norm,
            relu=relu)
    return layer


def weights_init(m):
    """
        Taken from https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
