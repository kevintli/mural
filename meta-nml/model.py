import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
import torch

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        # ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
        #     track_running_stats=False)),
        ('relu', nn.ReLU()),
        # ('pool', nn.MaxPool2d(2))
    ]))


class MetaConvModel(MetaModule):
    """4-layer Convolutional Neural Network architecture from [1].

    Parameters
    ----------
    in_channels : int
        Number of channels for the input images.

    out_features : int
        Number of classes (output of the model).

    hidden_size : int (default: 64)
        Number of channels in the intermediate representations.

    feature_size : int (default: 64)
        Number of features returned by the convolutional head.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_channels, out_features, hidden_size=64, feature_size=64):
        super(MetaConvModel, self).__init__()
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits


class MetaToyConvModel(MetaModule):
    def __init__(self, out_features, in_channels=1, hidden_size=64, feature_size=64):
        super(MetaToyConvModel, self).__init__()

        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            # ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
            #                       stride=1, padding=1, bias=True)),
            # ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
            #                       stride=1, padding=1, bias=True))
        ]))
        self.classifier = MetaLinear(feature_size, out_features, bias=True)

    def forward(self, inputs, params=None):
        inputs = torch.reshape(inputs, (-1, 1, 84, 84))
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

    def embedding(self, inputs, params=None):
        if type(inputs) == np.ndarray:
            inputs = torch.from_numpy(inputs)
        inputs = torch.reshape(inputs, (-1, 1, 100, 100))
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        return features


class MetaMNISTConvModel(MetaModule):
    def __init__(self, out_features, in_width=28, in_channels=1, hidden_size=32, mid_feats=512, feature_size=25088):
        super(MetaMNISTConvModel, self).__init__()

        self.in_width = in_width
        self.in_channels = in_channels
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.feature_size = in_width * in_width * hidden_size
        self.mid_feats = mid_feats

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
        ]))
        self.classifier_first = MetaLinear(self.feature_size, mid_feats, bias=True)
        self.classifier = MetaLinear(mid_feats, out_features, bias=True)

    def forward(self, inputs, params=None):
        inputs = torch.reshape(inputs, (-1, self.in_channels, self.in_width, self.in_width) )
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.reshape((features.size(0), -1))
        mid_logits = self.classifier_first(features, params=self.get_subdict(params, 'classifier_first'))
        logits = self.classifier(mid_logits, params=self.get_subdict(params, 'classifier'))
        return logits
    
    def embedding(self, inputs, params=None):
        inputs = torch.reshape(inputs, (-1, self.in_channels, self.in_width, self.in_width) )
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        features = features.view((features.size(0), -1))
        mid_logits = self.classifier_first(features, params=self.get_subdict(params, 'classifier_first'))
        return mid_logits

class MNISTConvModel(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class Flatten(nn.Module):
    def forward(self, input):
        # print(input.shape)
        return input.reshape(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 20, 7, 7)


class VAE(nn.Module):

    def __init__(self, in_width=28, z_dim=20, img_channels=1, h_dim=980):
        super(VAE, self).__init__()
        # in = img_channels x in_width x in_width
        ## encoder
        self.in_width = in_width
        self.img_channels = img_channels

        def conv_output_dim(input_size, kernel_size, stride=1, padding=0, **kwargs):
            from math import floor
            return floor((input_size + 2 * padding - (kernel_size - 1) - 1) / stride + 1)

        def conv_transpose_output_dim(input_size, kernel_size, stride=1, padding=0, dilation=1, **kwargs):
            return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

        # (H −1)×stride[0]−2×padding[0]+dilation[0]×(kernel_size[0]−1)+output_padding[0]+1

        ## encoder
        input_size = in_width
        conv1_filters = 10
        conv1_kwargs = dict(out_channels=10, kernel_size=3, stride=1, padding=1)
        a1_size = conv_output_dim(input_size, **conv1_kwargs)

        conv2_filters = 10
        conv2_kwargs = dict(out_channels=10, kernel_size=4, stride=2, padding=1)
        a2_size = conv_output_dim(a1_size, **conv2_kwargs)

        conv3_filters = 20
        conv3_kwargs = dict(out_channels=20, kernel_size=5, stride=2, padding=2)
        a3_size = conv_output_dim(a2_size, **conv2_kwargs)

        h_dim = a3_size ** 2 * conv3_filters
        print(a3_size)
        print(h_dim)

        ## decoder
        deconv1_filters = 10
        deconv1_kwargs = dict(kernel_size=5, stride=2, padding=2)
        d1_size = conv_transpose_output_dim(a3_size, **deconv1_kwargs)

        deconv2_filters = 10
        deconv2_kwargs = dict(kernel_size=5, stride=2, padding=1)
        d2_size = conv_transpose_output_dim(d1_size, **deconv2_kwargs)

        deconv3_filters = 20
        deconv3_kwargs = dict(kernel_size=6, stride=1, padding=2)
        d3_size = conv_transpose_output_dim(d2_size, **deconv3_kwargs)

        print(d1_size, d2_size, d3_size)

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channels, **conv1_kwargs),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv1_filters, **conv2_kwargs),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(conv2_filters, **conv3_kwargs),
            nn.ReLU()
        )
        self.to_dense = Flatten()
        ## map to latent z
        self.fc11 = nn.Linear(h_dim, z_dim)
        self.fc12 = nn.Linear(h_dim, z_dim)

        ## decoder
        self.fc2 = nn.Linear(z_dim, h_dim)
        self.reshape = UnFlatten()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(10, 10, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(10, img_channels, kernel_size=6, stride=1, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        a1 = self.conv1(x)
        a2 = self.conv2(a1)
        a3 = self.conv3(a2)
        h = self.to_dense(a3)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.reshape(self.fc2(z))
        a1 = self.deconv1(h)
        a2 = self.deconv2(a1)
        a3 = self.deconv3(a2)
        return a3

    def forward(self, x):
        x = torch.reshape(x, (-1, self.img_channels, self.in_width, self.in_width))
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class MetaMLPModel(MetaModule):
    """Multi-layer Perceptron architecture from [1].

    Parameters
    ----------
    in_features : int
        Number of input features.

    out_features : int
        Number of classes (output of the model).

    hidden_sizes : list of int
        Size of the intermediate representations. The length of this list
        corresponds to the number of hidden layers.

    References
    ----------
    .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
           for Fast Adaptation of Deep Networks. International Conference on
           Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
    """
    def __init__(self, in_features, out_features, hidden_sizes):
        super(MetaMLPModel, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_sizes = hidden_sizes

        layer_sizes = [in_features] + hidden_sizes
        self.features = MetaSequential(OrderedDict([('layer{0}'.format(i + 1),
            MetaSequential(OrderedDict([
                ('linear', MetaLinear(hidden_size, layer_sizes[i + 1], bias=True)),
                ('relu', nn.ReLU())
            ]))) for (i, hidden_size) in enumerate(layer_sizes[:-1])]))
        self.classifier = MetaLinear(hidden_sizes[-1], out_features, bias=True)

    def forward(self, inputs, params=None):
        features = self.features(inputs, params=self.get_subdict(params, 'features'))
        logits = self.classifier(features, params=self.get_subdict(params, 'classifier'))
        return logits

def ModelConvOmniglot(out_features, hidden_size=64):
    return MetaConvModel(1, out_features, hidden_size=hidden_size,
                         feature_size=hidden_size)

def ModelConvMiniImagenet(out_features, hidden_size=64):
    return MetaConvModel(3, out_features, hidden_size=hidden_size,
                         feature_size=5 * 5 * hidden_size)

def ModelMLPSinusoid(hidden_sizes=[40, 40]):
    return MetaMLPModel(1, 1, hidden_sizes)

def ModelMLPToy2D(hidden_sizes=[1024, 1024]):
    return MetaMLPModel(2, 2, hidden_sizes)

if __name__ == '__main__':
    model = ModelMLPToy2D()
