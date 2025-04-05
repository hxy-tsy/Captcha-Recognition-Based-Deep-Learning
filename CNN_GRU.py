import torch
import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(self, output_size):
        super(CNN_GRU,self).__init__()
        self.feature_extractor = QLNet()
        self.num_layers = 2
        self.n_directions = 2
        self.hidden_size = 80
        self.gru = nn.GRU(input_size=128, hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * self.n_directions, output_size)
        self.log_softmax = nn.LogSoftmax(2)
    def forward(self,input):
        x = self.feature_extractor(input)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        hidden = torch.zeros((self.num_layers * self.n_directions,
                             x.size(0), self.hidden_size),device='cpu')

        output, hidden = self.gru(x, hidden)
        x = self.fc(output)
        x = x.permute(1, 0, 2)
        output = self.log_softmax(x)
        output_lengths = torch.full(size=(x.size(1),), fill_value=x.size(0), dtype=torch.long,device='cpu')
        return output, output_lengths




class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1,1,0,bias=False)
        self.bn1=nn.BatchNorm2d(in_channels//2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 3, 1, 1,bias=False)
        self.bn2=nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += input
        x = self.relu(x)
        return x


class QLNet(nn.Module):
    def __init__(self):
        super(QLNet, self).__init__()
        self.conv1 = self._make_convolutional(3, 32, 3, 1, 1)
        self.conv2 =self._make_convolutional(32, 64, 3, 2, 1)
        self.layer1 = self._make_layer(64, 1)
        self.conv3 = self._make_convolutional(64, 128, 3, 2, 1)
        self.layer2 = self._make_layer(128, 2)
#         self.conv4 = _make_convolutional(128, 256, 3,2, 1)
#         self.layer3 = self._make_layer(256, 1)
#         self.conv5 = _make_convolutional(256, 512, 3, 2, 1)
#         self.layer4 = self._make_layer(512, 1)
#         self.conv6 = _make_convolutional(512, 1024, 3, (2,1), 1)
#         self.layer5 = self._make_layer(1024, 1)
        self.adapt_max_pool2d = nn.AdaptiveMaxPool2d((1, 40))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')

    def _make_layer(self, in_channels, repeat_count):
        layers = []
        for _ in range(repeat_count):
            layers.append(ResidualBlock(in_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.layer1(x)
        x = self.conv3(x)
        x = self.layer2(x)
#         x = self.conv4(x)
#         x = self.layer3(x)
#         x = self.conv5(x)
#         x = self.layer4(x)
#         x = self.conv6(x)
#         x = self.layer5(x)
        x=self.adapt_max_pool2d(x)
        return x

    def _make_convolutional(self,in_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )