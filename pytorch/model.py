import torch
import torch.nn as nn

class AlexNetModel(nn.Module):
    def __init__(self, LRN_depth=5, LRN_bias=2, LRN_alpha=0.0001, LRN_beta=0.75, drop_rate=0.2):
        super(AlexNetModel, self).__init__()
        self.LRN_depth = LRN_depth
        self.LRN_bias = LRN_bias
        self.LRN_alpha = LRN_alpha
        self.LRN_beta = LRN_beta

        ###### 1st conv_gpu1
        self.W1_1 = self.conv(3, 48, 11, 4, conv_padding=0, LRN=True, pooling=True)
        ###### 1st conv_gpu2
        self.W1_2 = self.conv(3, 48, 11, 4, conv_padding=0, LRN=True, pooling=True)

        ###### 2st conv_gpu1
        self.W2_1 = self.conv(48, 128, 5, 1, conv_padding=2, LRN=True, pooling=True)
        ###### 2st conv_gpu2
        self.W2_2 = self.conv(48, 128, 5, 1, conv_padding=2, LRN=True, pooling=True)

        ##### 3st conv
        self.W3 = self.conv(256, 384, 3, 1, conv_padding=1, LRN=False, pooling=False)

        ##### 4st conv gpu1
        self.W4_1 = self.conv(192, 192, 3, 1, conv_padding=1, LRN=False, pooling=False)
        ##### 4st conv gpu2
        self.W4_2 = self.conv(192, 192, 3, 1, conv_padding=1, LRN=False, pooling=False)

        ##### 5st conv gpu1
        self.W5_1 = self.conv(192, 128, 3, 1, conv_padding=1, LRN=False, pooling=True)
        ##### 5st conv gpu2
        self.W5_2 = self.conv(192, 128, 3, 1, conv_padding=1, LRN=False, pooling=True)

        ##### 1st fc
        self.W6 = nn.Sequential(nn.Linear(6 * 6 * 256, 4096), nn.Dropout(drop_rate), nn.ReLU())
        ##### 2st fc
        ##### this is original #####
        # self.W6 = nn.Sequential(nn.Linear(4096, 4096), nn.Dropout(drop_rate), nn.ReLU())
        ############################
        self.W7 = nn.Sequential(nn.Linear(4096, 1000), nn.Dropout(drop_rate), nn.ReLU())

        ##### 3st fc
        ##### this is original #####
        # self.7 = nn.Linear(4096, 2)
        ############################
        self.W8 = nn.Linear(1000, 2)

        nn.init.normal_(self.W1_1[0].weight, std=0.01)
        nn.init.normal_(self.W1_2[0].weight, std=0.01)
        nn.init.normal_(self.W2_1[0].weight, std=0.01)
        nn.init.normal_(self.W2_2[0].weight, std=0.01)
        nn.init.normal_(self.W3[0].weight, std=0.01)
        nn.init.normal_(self.W4_1[0].weight, std=0.01)
        nn.init.normal_(self.W4_2[0].weight, std=0.01)
        nn.init.normal_(self.W5_1[0].weight, std=0.01)
        nn.init.normal_(self.W5_2[0].weight, std=0.01)
        nn.init.normal_(self.W6[0].weight, std=0.01)
        nn.init.normal_(self.W7[0].weight, std=0.01)
        nn.init.normal_(self.W8.weight, std=0.01)

        nn.init.constant_(self.W1_1[0].bias, 0)
        nn.init.constant_(self.W1_2[0].bias, 0)
        nn.init.constant_(self.W2_1[0].bias, 1)
        nn.init.constant_(self.W2_2[0].bias, 1)
        nn.init.constant_(self.W3[0].bias, 0)
        nn.init.constant_(self.W4_1[0].bias, 1)
        nn.init.constant_(self.W4_2[0].bias, 1)
        nn.init.constant_(self.W5_1[0].bias, 1)
        nn.init.constant_(self.W5_2[0].bias, 1)
        nn.init.constant_(self.W6[0].bias, 1)
        nn.init.constant_(self.W7[0].bias, 1)
        nn.init.constant_(self.W8.bias, 1)

    def forward(self, x):
        ###### 1st conv_gpu1
        self.L1_1 = self.W1_1(x)  # [None, 48, 27, 27]
        ###### 1st conv_gpu2
        self.L1_2 = self.W1_2(x)  # [None, 48, 27, 27]

        ###### 2st conv_gpu1
        self.L2_1 = self.W2_1(self.L1_1)  # [None, 128, 13, 13]
        ###### 2st conv_gpu2
        self.L2_2 = self.W2_2(self.L1_2)  # [None, 128, 13, 13]

        ###### concat 2 gpu way before 3st conv
        self.L2 = torch.cat([self.L2_1, self.L2_2], dim=1)  # [None, 256, 13, 13]

        ##### 3st conv
        self.L3 = self.W3(self.L2) # [None, 384, 13, 13]

        ##### split into 2 way before 4st conv
        self.L3_1, self.L3_2 = torch.split(self.L3, split_size_or_sections=192, dim=1)  # [None, 192, 13, 13]

        ##### 4st conv gpu1
        self.L4_1 = self.W4_1(self.L3_1)  # [None, 192, 13, 13]
        ##### 4st conv gpu2
        self.L4_2 = self.W4_2(self.L3_2)  # [None, 192, 13, 13]

        ##### 5st conv gpu1
        self.L5_1 = self.W5_1(self.L4_1)  # [None, 128, 6, 6]
        ##### 5st conv gpu2
        self.L5_2 = self.W5_2(self.L4_2)  # [None, 128, 6, 6]

        ###### concat 2 gpu way before 1st fcl
        self.L5 = torch.cat([self.L5_1, self.L5_2], dim=1)  # [None, 256, 6, 6]
        ###### and flatten
        self.L5 = self.L5.view(-1, 256 * 6 * 6)

        ##### 1st fc
        self.L6 = self.W6(self.L5)  # [None, 6 * 6 * 256]

        ##### 2st fc
        self.L7 = self.W7(self.L6)  # [None, 1000]

        ##### 3st fc
        self.logit = self.W8(self.L7)  # [None, 2]

        return self.logit

    def conv(self, input_ch, output_ch, kernel_size, stride, conv_padding=0, LRN=False, pooling=False, activation=nn.ReLU()):
        modules = []

        modules.append(nn.Conv2d(input_ch, output_ch, kernel_size=kernel_size, stride=stride, padding=conv_padding))
        if LRN:
            modules.append(nn.LocalResponseNorm(size=self.LRN_depth, k=self.LRN_bias, alpha=self.LRN_alpha, beta=self.LRN_beta))
        if pooling:
            modules.append(nn.MaxPool2d(kernel_size=3, stride=2))
        if activation is not None:
            modules.append(activation)

        return nn.Sequential(*modules)