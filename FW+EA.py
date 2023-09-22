'''
文件用途：固定窗测试
作者：陈欣如
日期：2022年09月16日
python FW+EA.py --seed 2022 --model_num 2 --train_length 100 --dataset '001-2014' --classes 4 --person 1 --model_type 'EEGNet' --model_save_path /model/EEGNet-001-2014-4/cross_overlap/ --gpu_id '3' --modelEA_save_path /model/EEGNet-001-2014-4/cross_overlap_EA/
'''
import numpy as np
import os
import joblib
from scipy import signal
import warnings
from sklearn.ensemble import VotingRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import mne
import scipy.io as scio
from xlwt import *
warnings.filterwarnings("ignore")
import argparse
import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
from scipy.io import loadmat
from utils import EA




class PatchEmbedding(nn.Module):
    def __init__(self,channel):
        # self.patch_size = patch_size
        super().__init__()

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
                kernel_size=(channel, 1),  # filter size #bonn数据集单通道c=1
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            # nn.Dropout(0.25)  # output shape (16, 1, T//4)
            # Rearrange('b e (h) (w) -> b (h w) e'),
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
            # nn.Conv2d(
            #     in_channels=16,  # input shape (16, 1, T//4)
            #     out_channels=16,  # num_filters
            #     kernel_size=(1, 1),  # filter size
            #     bias=False
            # ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            # nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = F.dropout(x, 0.25)
        x = self.block_3(x)
        x = F.dropout(x, 0.25)
        # print(x.shape)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=4,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        # x = x.contiguous().view(x.size(0), -1)
        out = self.clshead(x)
        return out


class Conformer(nn.Sequential):
    def __init__(self, channel,emb_size=16, depth=3, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(channel),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


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
            # nn.Dropout(self.drop_out)
        )
        self.block_1.eval()
        self.block_2.eval()
        self.block_3.eval()
        out = self.block_1(torch.zeros(1, 1, input_ch, input_time))
        out = self.block_2(out)
        out = self.block_3(out)
       # n_out_time = out.cpu().data.numpy().shape[3]
       # self.final_conv_length = n_out_time
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Linear(self.n_outputs, self.n_classes)


    def forward(self, x):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        x = F.dropout(x, 0.25)
        # print(x.shape)
        x = self.block_3(x)
        x = F.dropout(x, 0.25)
        x = x.view(x.size()[0], -1)
        x = self.clf(x)
        # print(x)
        # print(F.softmax(x, dim=1))


        return x #返回结果


#得到结果 0123
class get_result():
    def __init__(self,model_number,length,seedval):
        super().__init__()
        self.model_num = model_number
        self.cal_len = length
        self.seeder=seedval

    def band_Filter(self, data, bandID):
        data = signal.detrend(data, axis=-1, type='linear', )  # 去趋势化
        b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
        filtedData = signal.filtfilt(b, a, data, axis=1)
        return filtedData

    def getModel(self, personID,C,model_file):

        model = {}  # 创建当前用户的模型字典
        for i in range(self.model_num):
            seed = self.seeder + i
            model_path = current_working_dir + model_file + str(personID) +'_' + str(seed) + '.pth'
            if args.model_type=='EEGNet':
                model[i] = EEGNet(class_num, C, self.cal_len, 0.25)
                model[i].load_state_dict(torch.load(model_path, map_location='cpu'))
            elif args.model_type=='Transformer':
                model[i] = Conformer(channel=C,emb_size=16, depth=3, n_classes=class_num)
                model[i].load_state_dict(torch.load(model_path, map_location='cpu'))
        return model

    def recognize(self, data, personID,model_file):
        if args.model_type=='EEGNet': 
            data = self.band_Filter(data, personID)
            C=data.shape[0]
            data = np.expand_dims(data, 0)
            data = np.expand_dims(data, 0)
            data = torch.from_numpy(data.copy()).to(torch.float32)
            mod = self.getModel(personID,C,model_file)
            ## mean_y 是加权平均预测结果，vote_y是投票方式确定
            for i in range(self.model_num):
                self.set_seed(i + 2022)
                mod[i].eval()
                output = mod[i](data)
                if i == 0:
                    mean_y = torch.softmax(output, dim=1)
                else:
                    mean_y += torch.softmax(output, dim=1)
            mean_y /= self.model_num
            max_pre=torch.max(mean_y).data.numpy()
            pred_y_mean = torch.max(mean_y.cpu(), 1)[1].data.numpy()
            result = pred_y_mean
            return result,max_pre
        
        elif args.model_type=='Transformer':
            data = self.band_Filter(data, personID)
            data = np.expand_dims(data, 0)
            data = np.einsum('acd, ce -> aed', data, Wb)
            data = np.expand_dims(data, 0)
            C=data.shape[2]
            data = torch.from_numpy(data.copy()).to(torch.float32)
            mod = self.getModel(personID,C,model_file)
            ## mean_y 是加权平均预测结果，vote_y是投票方式确定
            for i in range(self.model_num):
                self.set_seed(i + 2022)
                mod[i].eval()
                output = mod[i](data)
                if i == 0:
                    mean_y = torch.softmax(output, dim=1)
                else:
                    mean_y += torch.softmax(output, dim=1)
            mean_y /= self.model_num
            max_pre=torch.max(mean_y).data.numpy()
            pred_y_mean = torch.max(mean_y.cpu(), 1)[1].data.numpy()
            result = pred_y_mean
            return result,max_pre
        elif args.model_type=='CSP+SVM':
            data = self.band_Filter(data, personID)
            data = np.expand_dims(data, 0)
            mod = joblib.load(current_working_dir +model_file + str(personID) + ".m")
            csp_data=mod[0].transform(data)
            result = mod[1].predict(csp_data)
            max_pre= np.max(mod[1].predict_proba(csp_data))
            return result,max_pre
           

    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

#计算itr
def itr_cal(itr_acc,itr_data_len,type_class):
    itr = 60*((np.log2(type_class)+itr_acc*np.log2(itr_acc)) + (1-itr_acc)*np.log2((1-itr_acc)/(type_class-1)))/(int(itr_data_len)/250)
    return itr

if __name__ == '__main__':

    current_working_dir = os.getcwd()
    
    parser = argparse.ArgumentParser(description='MI调参')
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_num', type=int, default=11, help="ensemble models")
    parser.add_argument('--train_length', type=int, default=100, help="Fixed window length for training")
    parser.add_argument('--dataset', type=str, default='001-2014',choices=['001-2014', '004-2014'],help="Choose a dataset you want to explore")
    parser.add_argument('--classes', type=int,default=2, choices=[2,4],help="N classes")
    parser.add_argument('--person', type=int, default=1 ,help="Choose the person you want to explore")
    parser.add_argument('--model_type', type=str, default='EEGNet' ,choices=['EEGNet', 'Transformer','CSP+SVM'],help="Choose a backbone you want to explore")
    parser.add_argument('--model_save_path', type=str,default='/model/' ,help="model_save_path")
    parser.add_argument('--modelEA_save_path', type=str,default='/model/' ,help="model_save_path with EA")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

    args = parser.parse_args()

    rootdir = os.path.dirname(os.path.abspath(__file__))

    Fstop1=8
    Fstop2=26
    fs = 250  
    MItime=3

    class_num=args.classes
    personid=args.person
    data_path = current_working_dir + "/datasets/"+args.dataset+"/"
    if args.model_type=='Transformer':
        Wb= np.load(current_working_dir + args.model_save_path + str(personid) +  '_csp.npy')


    if args.dataset == "001-2014" and class_num == 4:
        print('person==%d' % personid)
        path1 = data_path + "A0" + str(personid) + "E.gdf"
        rawDataGDF = mne.io.read_raw_gdf(path1, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right'])
        event_position = rawDataGDF.annotations.onset  
        event_type = rawDataGDF.annotations.description 
        temp = rawDataGDF.to_data_frame().drop(['time'], axis=1)
        chan_time_all = temp.T.to_numpy()

       
        label_path = data_path + "A0" + str(personid) + "E.mat"
        true_labels = scio.loadmat(label_path)

        correct_num = 0
        total_num = 0
        data_save=[]
        result_go = get_result(model_number=args.model_num, length=args.train_length, seedval=args.seed)

        for xuhao, type_mi in enumerate(event_type):
            if type_mi == '783':
                total_num = total_num+1
                if total_num<=10:
                    event_start_position = int(event_position[xuhao] * fs)
                    data_what= (
                    chan_time_all[:, (event_start_position + fs):(event_start_position + fs + args.train_length)])
                    data_save.append(data_what)
                    result, max_pre_what = result_go.recognize(data=data_what, personID=personid,model_file=args.model_save_path)
                    result = int(result)  # 0123
                    label_now = int(np.array(true_labels['classlabel'][total_num - 1]))
                    if (result + 1) == label_now:
                        correct_num = correct_num + 1
                if total_num==10:
                    data_save = np.array(data_save)
                    data_save,sqrtRefEA = EA.EA(data_save)


                if total_num>=10:


                    event_start_position = int(event_position[xuhao] * fs)
                    data_what = (chan_time_all[:,
                                    (event_start_position + fs):(event_start_position + fs + args.train_length)])
                    data_what = np.dot(sqrtRefEA, data_what)
                    result, max_pre_what = result_go.recognize(data=data_what, personID=personid,
                                                                    model_file=args.modelEA_save_path)
                    result = int(result)  # 0123
                    label_now = int(np.array(true_labels['classlabel'][total_num - 1]))

                    if (result + 1) == label_now:
                        correct_num = correct_num + 1



        acc = np.round(np.array(correct_num/total_num), decimals=4)
        finnal_ITR = itr_cal(acc, args.train_length, class_num)
        print(np.round(np.array(acc), decimals=4))
        print(np.round(np.array(finnal_ITR), decimals=4))


    elif args.dataset == "001-2014" and class_num == 2:
        print('person==%d' % personid)
        path1 = data_path + "A0" + str(personid) + "E.gdf"
        rawDataGDF = mne.io.read_raw_gdf(path1, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right'])
        event_position = rawDataGDF.annotations.onset  
        event_type = rawDataGDF.annotations.description 
        temp = rawDataGDF.to_data_frame().drop(['time'], axis=1)
        chan_time_all = temp.T.to_numpy()

       
        label_path = data_path + "A0" + str(personid) + "E.mat"
        true_labels = scio.loadmat(label_path)

        correct_num = 0
        total_num = 0
        count=0
        data_save=[]
        result_go = get_result(model_number=args.model_num, length=args.train_length, seedval=args.seed)

        for xuhao, type_mi in enumerate(event_type):
            if type_mi == '783':
                label_now = int(np.array(true_labels['classlabel'][count]))
                count = count+1
                if label_now==1 or label_now==2:
                    total_num = total_num+1
                    if total_num<=10:
                        event_start_position = int(event_position[xuhao] * fs)
                        data_what= (
                        chan_time_all[:, (event_start_position + fs):(event_start_position + fs + args.train_length)])
                        data_save.append(data_what)
                        result, max_pre_what = result_go.recognize(data=data_what, personID=personid,model_file=args.model_save_path)
                        result = int(result)  # 0123
                        if (result + 1) == label_now:
                            correct_num = correct_num + 1
                    if total_num==10:
                        data_save = np.array(data_save)
                        data_save,sqrtRefEA = EA.EA(data_save)


                    if total_num>=10:


                        event_start_position = int(event_position[xuhao] * fs)
                        data_what = (chan_time_all[:,
                                        (event_start_position + fs):(event_start_position + fs + args.train_length)])
                        data_what = np.dot(sqrtRefEA, data_what)
                        result, max_pre_what = result_go.recognize(data=data_what, personID=personid,
                                                                        model_file=args.modelEA_save_path)
                        result = int(result)  # 0123

                        if (result + 1) == label_now:
                            correct_num = correct_num + 1



        acc = np.round(np.array(correct_num/total_num), decimals=4)
        finnal_ITR = itr_cal(acc, args.train_length, class_num)
        print(np.round(np.array(acc), decimals=4))
        print(np.round(np.array(finnal_ITR), decimals=4))


    elif args.dataset == "004-2014" and class_num == 2:

        file_path = data_path + "B0" + str(personid) + "E.mat"
        print('person==%d' % personid)
        all_data = loadmat(file_path)
        block_nums = 2  
        #记录正确个数以及总个数
        correct_num = 0
        total_num = 0
        data_save = []

        result_go = get_result(model_number=args.model_num, length=args.train_length, seedval=args.seed)


        X = all_data["data"][0, 0]["X"][0, 0][:, :3].T  # X(point,channle) data
        Y = all_data["data"][0, 0]["y"][0, 0][:, 0]  # y(0,trial) class
        TRIAL = all_data["data"][0, 0]["trial"][0, 0][:, 0]
        for xuhao, event_position in enumerate(TRIAL):
            total_num = total_num + 1
            if total_num <= 10:
                event_start_position = int((event_position - 1) + fs * 4)
                data_what_all = X[:, (event_start_position):(event_start_position + args.train_length)]
                result ,max_pre_what= result_go.recognize(data=data_what_all,personID=personid,model_file=args.model_save_path)
                data_save.append(data_what_all)
                label_now = int(np.array(Y[xuhao]))
                if result + 1 == label_now:
                    correct_num = correct_num + 1
               
            if total_num==10:
                data_save = np.array(data_save)
                data_save,sqrtRefEAT = EA.EA(data_save)
            if total_num >= 10:
                event_start_position = int((event_position - 1) + fs * 4)
                data_what_all = X[:, (event_start_position):(event_start_position + args.train_length)]
                data_what_all = np.dot(sqrtRefEAT, data_what_all)
                result, max_pre_what = result_go.recognize(data=data_what_all, personID=personid,
                                                            model_file=args.modelEA_save_path)
                label_now = int(np.array(Y[xuhao]))
                if result + 1 == label_now:
                    correct_num = correct_num + 1
               



        X = all_data["data"][0, 1]["X"][0, 0][:, :3].T  # X(point,channle) data
        Y = all_data["data"][0, 1]["y"][0, 0][:, 0]  # y(0,trial) class
        TRIAL = all_data["data"][0, 1]["trial"][0, 0][:, 0]
        for xuhao, event_position in enumerate(TRIAL):

            total_num = total_num + 1
            event_start_position = int((event_position - 1) + fs * 4)
            data_what_all = X[:, (event_start_position):(event_start_position + args.train_length)]
            data_what_all = np.dot(sqrtRefEAT, data_what_all)
            result, max_pre_what = result_go.recognize(data=data_what_all, personID=personid,
                                                        model_file=args.modelEA_save_path)
            label_now = int(np.array(Y[xuhao]))
            if result + 1 == label_now:
                correct_num = correct_num + 1
          



        acc = correct_num / total_num
        if acc <= 0.5:
            finnal_ITR = 0
        else:
            finnal_ITR = itr_cal(acc, args.train_length, class_num)

        print(np.round(np.array(acc), decimals=4))
        print(np.round(np.array(finnal_ITR), decimals=4))
