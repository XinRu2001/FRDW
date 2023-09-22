'''
文件用途：eegnet的model
作者：陈欣如
日期：2022年01月17日
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, classes_num,input_ch, input_time,dropout_size):
        super(EEGNet, self).__init__()
        self.drop_out = dropout_size
        self.n_classes=classes_num
        #within-subjet=0.5 cross-subject=0.25

        #block1 is the common conv
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d(padding=(31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters,F1=8
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 is implementation of Depthwise Convolution
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(input_ch, 1),  # filter size #bonn数据集单通道c=1
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            # nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        # block 3 is implementation of  Separable Convolution
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
        )
        self.block_1.eval()
        self.block_2.eval()
        self.block_3.eval()
        out = self.block_1(torch.zeros(1, 1, input_ch, input_time))
        out = self.block_2(out)
        out = self.block_3(out)
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Linear(self.n_outputs, self.n_classes)


    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = F.dropout(x, 0.25)
        x = self.block_3(x)
        x = F.dropout(x, 0.25)
        x = x.view(x.size()[0], -1)
        x = self.clf(x)


        return x #返回结果




if __name__ == '__main__':
    net = EEGNet(classes_num=4, input_ch=22, input_time=750,dropout_size=0.25)
    data = torch.rand(16, 1, 22, 750)
    output = net(data)
    print(output.shape)