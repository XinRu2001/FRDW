'''
get training data and lable
'''
from scipy.io import loadmat
import numpy as np
import os,sys
import mne
from scipy import signal


#########################preprocessing########################################
def preprocessing(data,fs):
    # 设置滤波范围
    Fstop1 = 8
    Fstop2 = 26
    # 通道选择方式
    filtedData = data
    filtedData = signal.detrend(filtedData, axis=-1, type='linear', )  # 去趋势化
    # 滤波
    b, a = signal.butter(6, [2.0 * Fstop1 / fs, 2.0 * Fstop2 / fs], 'bandpass')  # 5阶巴特沃斯滤波器
    filtedData = signal.filtfilt(b, a, filtedData, axis=1)

    return filtedData

#########################preprocessing########################################

###########################004-2014###################################
def data_class_004(position,all_data,len,fs):
    X = all_data["data"][0, position]["X"][0,0][:,:3].T # X(point,channle) data
    Y = all_data["data"][0, position]["y"][0,0][:,0] # y(0,trial) class
    TRIAL=all_data["data"][0, position]["trial"][0,0][:,0] #trial(0,position)从1开始，因此用时要-1
    data_class1 = []
    data_class2 = []
    X = preprocessing(X,fs)
    for xuhao, y in enumerate(Y):
        if y == 1:
            data=X[:, int((TRIAL[xuhao]-1)+fs*4):int((TRIAL[xuhao]-1)+fs*(len+4))]
            data_class1.append(data)

        elif y == 2:
            data=X[:, int((TRIAL[xuhao]-1)+fs*4):int((TRIAL[xuhao]-1)+fs*(len+4))]
            data_class2.append(data)

    return data_class1,data_class2 #[seg,chanle,point] 几段数据，通道，数据点



def get_data_train_004(path):
    all_data_train = loadmat(path)
    segment_len_s = 3 #只要前三秒的数据
    fs=250
    #class 1:'right hand' 2:'feet'
    seg_len = 3 #三段
    data_1=[]
    data_2=[]
    for i in range(seg_len):
        data_class1,data_class2 = data_class_004(i,all_data_train,segment_len_s,fs)
        if i == 0:
            data_1 = data_class1
            data_2 = data_class2
        else:
            data_1.extend(data_class1)
            data_2.extend(data_class2)

    return data_1,data_2

###########################004-2014###################################

###########################001-2014-4###################################
def get_data_train_001_4(path):
    rawDataGDF1 = mne.io.read_raw_gdf(path, preload=True, exclude=['EOG-left', 'EOG-central', 'EOG-right'])
    event_position1 = rawDataGDF1.annotations.onset  # 事件位置列表
    event_type1 = rawDataGDF1.annotations.description  # 事件名称
    temp1 = rawDataGDF1.to_data_frame().drop(['time'], axis=1)
    chan_time_all1 = temp1.T.to_numpy()
    fs=250
    # 滤波
    pre_data1 = preprocessing(chan_time_all1,fs)

    # 分段
    mi_left = []
    mi_right = []
    mi_foot = []
    mi_tongue = []

    for xuhao1, type_mi1 in enumerate(event_type1):
        if type_mi1 == '769':
            event_start_position1 = int(event_position1[xuhao1] * fs)
            mi_left.append(pre_data1[:, event_start_position1 + fs:event_start_position1 + fs * 4])
        elif type_mi1 == '770':
            event_start_position1 = int(event_position1[xuhao1] * fs)
            mi_right.append(pre_data1[:, event_start_position1 + fs:event_start_position1 + fs * 4])
        elif type_mi1 == '771':
            event_start_position1 = int(event_position1[xuhao1] * fs)
            mi_foot.append(pre_data1[:, event_start_position1 + fs:event_start_position1 + fs * 4])
        elif type_mi1 == '772':
            event_start_position1 = int(event_position1[xuhao1] * fs)
            mi_tongue.append(pre_data1[:, event_start_position1 + fs:event_start_position1 + fs * 4])

    return mi_left,mi_right,mi_foot,mi_tongue

###########################001-2014-4###################################

###########################001-2014-2###################################
def get_data_train_001_2(path):
    rawDataGDF = mne.io.read_raw_gdf(path, preload=True,exclude=['EOG-left', 'EOG-central', 'EOG-right'])
    event_position = rawDataGDF.annotations.onset#事件位置列表
    event_type = rawDataGDF.annotations.description#事件名称
    temp = rawDataGDF.to_data_frame().drop(['time'], axis=1)
    chan_time_all = temp.T.to_numpy()
    fs=250
    #preprocessing
    pre_data=preprocessing(chan_time_all,fs)
    #segment class
    mi_left=[]
    mi_right=[]

    for xuhao,type_mi in enumerate(event_type):
        if type_mi == '769':
            event_start_position =int(event_position[xuhao]*fs)
            mi_left.append(pre_data[:,event_start_position+fs:event_start_position+fs*4])
        elif type_mi == '770':
            event_start_position =int(event_position[xuhao]*fs)
            mi_right.append(pre_data[:,event_start_position+fs:event_start_position+fs*4])

    return mi_left, mi_right
