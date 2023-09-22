'''
Find the best parameters and train the model
python within-subject_train.py --lr 0.001 --gpu_id '2' --seed 2022 --epoch 180 --bs 64 --train_len 100 --model_num 2 --dataset '001-2014' --classes 2 --person 1 --augmentation overlap --overlap 25 --model_type Transformer --model_save_path /model/try/
'''
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.utils.data as Data
import random
import argparse
from utils import get_data, EEGNet, Transformer,common_spatial_pattern
from mne.decoding import CSP
from sklearn.svm import SVC
import joblib



################Fixed randomness for easy reproduction############
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

################ For verification ############
def evaluate_validation(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)
    loss_func_2 = nn.CrossEntropyLoss().cuda()
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            logits = model.forward(x)
            pred_y = torch.max(logits, 1)[1]

            #true_y = y.data.detach().cpu().numpy().astype(int)
            correct += float((pred_y == y).sum())
    loss_val = loss_func_2(logits, y.long())
    val_acc = correct / total

    return val_acc,loss_val

################ dataloader ############
def torch_transform(TORCH_X,TORCH_Y):
    TORCH_X = np.expand_dims(TORCH_X, 1)

    X_train = torch.from_numpy(TORCH_X).to(torch.float32)

    y_train = torch.from_numpy(TORCH_Y)
  
    dataset = Data.TensorDataset(X_train, y_train)
    dataloader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,drop_last = True)
    return dataloader


################ For train and retrain############
def mod_train(X_TRAIN, Y_TRAIN, X_VALID, Y_VALID,model_save_path):
    print("=========find the best epoch===========")
 
    C=X_TRAIN.shape[1]
    
    dataloader_train = torch_transform(X_TRAIN, Y_TRAIN)
    dataloader_valid = torch_transform(X_VALID, Y_VALID)
    
    if args.model_type == "Transformer":
        mod = Transformer.Conformer(channel=C,emb_size=16, depth=3, n_classes=CLASSES_NUM).cuda()
    elif args.model_type == "EEGNet":    
        mod = EEGNet.EEGNet(classes_num= CLASSES_NUM, input_ch=C, input_time=train_len, dropout_size=0.25).cuda()
    optimizer = torch.optim.Adam(mod.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().cuda()
    print('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))
    
    best_acc = 0 
    best_epoch = 0
    train_correct = 0
    val_acc_all = [] 
    train_acc_all = []
    loss_item_train = []  
    loss_item_val = []
   
        
 
    for epoch in range(EPOCH):
        for step, (b_x, b_y) in enumerate(dataloader_train):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            mod.train()
            output = mod.forward(b_x)
            loss = loss_func(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(output, 1)[1]
            train_correct += float((pred == b_y).sum())

        print('Epoch[{}/{}],Loss:{:.4f}'.format(epoch + 1, EPOCH, loss.item()))
        loss_item_train.append(loss.item())  # 损失list
        train_acc = train_correct / len(dataloader_train.dataset)
        train_acc_all.append(train_acc)
        train_correct = 0
        mod.eval() 
        val_acc, val_loss = evaluate_validation(mod, dataloader_valid)
        print('valid acc of the model :{:.3f}'.format(val_acc))
        if (epoch - best_epoch) > 40:
            break
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            # torch.save(mod.state_dict(), model_save_path)
        val_acc_all.append(val_acc)  # 验证集准确率list
        loss_item_val.append(val_loss.item())

    print('epoch=%d, \t ,验证集最好的ACC: best_acc=%.5f.' % (int(best_epoch + 1), best_acc))
    

    print("=========retrain model===========")
 
    dataloader_train = torch_transform(np.vstack((X_TRAIN,X_VALID)), np.concatenate((Y_TRAIN,Y_VALID)))
    
    if args.model_type == "Transformer":
        mod = Transformer.Conformer(channel=C,emb_size=16, depth=3, n_classes=CLASSES_NUM).cuda()
    elif args.model_type == "EEGNet":    
        mod = EEGNet.EEGNet(classes_num= CLASSES_NUM, input_ch=C, input_time=train_len, dropout_size=0.25).cuda()
    optimizer = torch.optim.Adam(mod.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss().cuda()
    print('use {}'.format("cuda" if torch.cuda.is_available() else "cpu"))
   

    for epoch in range(best_epoch + 1):
        for step, (b_x, b_y) in enumerate(dataloader_train):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            mod.train()
            output = mod.forward(b_x)
            loss = loss_func(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(output, 1)[1]
            

        print('Epoch[{}/{}],Loss:{:.4f}'.format(epoch + 1, EPOCH, loss.item()))

        mod.eval()  
        if epoch == best_epoch:
            torch.save(mod.state_dict(), model_save_path)

    return loss_item_train, loss_item_val, train_acc_all, val_acc_all,best_epoch + 1,best_acc


################ Trial segmentation ############
def segment_pick(data,postdata_len,strip):
    seg_data = []
    data = np.array(data)
    lengh = data.shape[1]
    postdata_len = int(postdata_len)
    i = postdata_len
    while 1:
        if i > lengh:
            break
        seg_data.append(data[:, i-postdata_len:i])

        i = i + strip

    return seg_data

################ Data augmentation############
def augment(data,aug_type):
    if aug_type=="overlap" or aug_type=="FR":
        strip0 = args.overlap
    elif aug_type=="none":
        strip0 = train_len
    if aug_type=="overlap" or aug_type=="none":
        for xuhao, i in enumerate(data):
            data_oneaug = segment_pick(i, train_len, strip0) 
            if xuhao == 0:
                data_aug = data_oneaug
            else:
                data_aug = np.vstack((data_aug, data_oneaug))
    elif aug_type=="FR":
        pad_len = int(0.7 * train_len) #set 30% reproduction
        for xuhao, i in enumerate(data):
            data_oneaug = np.array(segment_pick(i, pad_len, strip0))  
            if xuhao == 0:
                data_oneaug = np.concatenate((data_oneaug, data_oneaug[:, :, :(train_len - pad_len)]), axis=-1)
                data_aug = data_oneaug
            else:
                data_oneaug = np.concatenate((data_oneaug, data_oneaug[:, :, :(train_len - pad_len)]), axis=-1)
                data_aug = np.vstack((data_aug, data_oneaug))

    return data_aug

################ calculate ITR############
def itr_cal(itr_acc,itr_data_len,type_class):
    itr = 60*((np.log2(type_class)+itr_acc*np.log2(itr_acc)) + (1-itr_acc)*np.log2((1-itr_acc)/(type_class-1)))/(int(itr_data_len)/250)
    return itr


################ To prepare training and testing data############
def data_and_label(dataset="004-2014"):
    if dataset =="004-2014":
        file_path = current_working_dir + "/datasets/004-2014/" + "B0" + str(personid) + "T.mat"
        data_train_1, data_train_2 = get_data.get_data_train_004(file_path)  

        data1_train = augment(data_train_1[:160], aug_type=args.augmentation)
        data2_train = augment(data_train_2[:160], aug_type=args.augmentation)
        data1_evl = augment(data_train_1[160:], aug_type=args.augmentation)
        data2_evl = augment(data_train_2[160:], aug_type=args.augmentation)

        data_X_train = np.vstack((data1_train, data2_train))
        data_X_evl = np.vstack((data1_evl, data2_evl))

        def Y_get(number):
            Y1 = np.full(shape=(number,), fill_value=0, dtype='float64')
            Y2 = np.full(shape=(number,), fill_value=1, dtype='float64')

            data_Y = np.concatenate((Y1, Y2))
            return data_Y

        data_Y_train = Y_get(len(np.array(data1_train)))
        data_Y_evl = Y_get(len(np.array(data1_evl)))

    elif dataset =="001-2014" and CLASSES_NUM==2:
        file_path = current_working_dir + "/datasets/001-2014/" + "A0" + str(personid) + "T.gdf"
        data_train_1, data_train_2 = get_data.get_data_train_001_2(file_path)
      
        

        data1_train = augment(data_train_1[:60], aug_type=args.augmentation)
        data2_train = augment(data_train_2[:60], aug_type=args.augmentation)

        data1_evl = augment(data_train_1[60:], aug_type=args.augmentation)
        data2_evl = augment(data_train_2[60:], aug_type=args.augmentation)

        data_X_train = np.vstack((data1_train, data2_train))
        data_X_evl = np.vstack((data1_evl, data2_evl))

        def Y_get(number):
            Y1 = np.full(shape=(number,), fill_value=0, dtype='float64')
            Y2 = np.full(shape=(number,), fill_value=1, dtype='float64')

            data_Y = np.concatenate((Y1, Y2))
            return data_Y

        data_Y_train = Y_get(len(np.array(data1_train)))
        data_Y_evl = Y_get(len(np.array(data1_evl)))

    elif dataset == "001-2014" and CLASSES_NUM == 4:
        file_path = current_working_dir + "/datasets/001-2014/" + "A0" + str(personid) + "T.gdf"
        mi_left,mi_right,mi_foot,mi_tongue = get_data.get_data_train_001_4(file_path)

        data1_train = augment(mi_left[:60], aug_type=args.augmentation)
        data2_train = augment(mi_right[:60], aug_type=args.augmentation)
        data3_train = augment(mi_foot[:60], aug_type=args.augmentation)
        data4_train = augment(mi_tongue[:60], aug_type=args.augmentation)

        data1_evl = augment(mi_left[60:], aug_type=args.augmentation)
        data2_evl = augment(mi_right[60:], aug_type=args.augmentation)
        data3_evl = augment(mi_foot[60:], aug_type=args.augmentation)
        data4_evl = augment(mi_tongue[60:], aug_type=args.augmentation)

        data_X_train = np.vstack((data1_train, data2_train,data3_train,data4_train))
        data_X_evl = np.vstack((data1_evl, data2_evl, data3_evl, data4_evl))


        def Y_get(number):
            Y769 = np.full(shape=(number,), fill_value=0, dtype='float64')
            Y770 = np.full(shape=(number,), fill_value=1, dtype='float64')
            Y771 = np.full(shape=(number,), fill_value=2, dtype='float64')
            Y772 = np.full(shape=(number,), fill_value=3, dtype='float64')
            data_Y = np.concatenate((Y769, Y770, Y771, Y772))
            return data_Y

        data_Y_train = Y_get(len(np.array(data1_train)))
        data_Y_evl = Y_get(len(np.array(data1_evl)))
    return data_X_train, data_Y_train, data_X_evl, data_Y_evl



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MI_find_hyper')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--seed', type=int, default=2022, metavar='S',
                        help='random seed (default: 2022)')
    parser.add_argument('--epoch', type=int, default=180,
                        help='max epoch')
    parser.add_argument('--bs', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--train_len', type=int, default=100,
                        help='Choose a data length you want to explore')
    parser.add_argument('--model_num', type=int, default=11, help="ensemble models")
    parser.add_argument('--dataset', type=str, default='001-2014',choices=['001-2014', '004-2014'],help="Choose a dataset you want to explore")
    parser.add_argument('--classes', type=int,default=2, choices=[2,4],help="N classes")
    parser.add_argument('--person', type=int, default=1 ,help="Choose the person you want to explore")
    parser.add_argument('--augmentation', type=str, default='none' ,choices=['none', 'overlap','FR'],help="Choose a augmentation method you want to explore")
    parser.add_argument('--overlap', type=int, default=25, help='overlap_points')
    parser.add_argument('--model_type', type=str, default='EEGNet' ,choices=['EEGNet', 'Transformer','CSP+SVM'],help="Choose a backbone you want to explore")
    parser.add_argument('--model_save_path', type=str,default='/model/' ,help="Choose a backbone you want to explore")

    args = parser.parse_args()

   
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    train_len=args.train_len
    personid=args.person
    CLASSES_NUM=args.classes
    current_working_dir = os.getcwd()
    
    torch.cuda.set_device('cuda:'+ args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id #gpu

    data_X_train, data_Y_train, data_X_evl, data_Y_evl = data_and_label(args.dataset)
    print("data_X_train.shape===",data_X_train.shape)
    print("data_X_evl.shape===",data_X_evl.shape)


    if args.model_type == "EEGNet":
        ###########Initial Parameters############
        LR=args.lr
        EPOCH =args.epoch
        BATCH_SIZE = args.bs
        model_num=args.model_num
        #########################################


        print("===================start training====================")
        itr_all=[]
        acc_all=[]
        for i in range(model_num):
            print('Model===%d'%(i+1))
            seedval = args.seed + i
            seed_torch(seedval)
            
            model_name = current_working_dir + args.model_save_path + str(personid) + '_' + str(seedval) + '.pth'
            losstrain,lossval,train_acc_all,val_acc_all,Best_EPOCH,Best_ACC=mod_train(X_TRAIN=data_X_train, Y_TRAIN=data_Y_train, X_VALID=data_X_evl, Y_VALID=data_Y_evl,model_save_path=model_name)
            itr_i=itr_cal(Best_ACC, train_len, CLASSES_NUM)
            itr_all.append(itr_i)
            acc_all.append(Best_ACC)
            print("validation ACC : {}, ITR : {} , best epoch : {}".format(Best_ACC,itr_i,Best_EPOCH))


    elif args.model_type == 'CSP+SVM':
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        X_train=csp.fit_transform(data_X_train, data_Y_train)
        X_evl = csp.transform(data_X_evl)

        clf = SVC(kernel = "linear",probability = True,random_state=2022,C=0.1).fit(X_train,data_Y_train)
        acc_all = clf.score(X_evl,data_Y_evl)
        itr_all=itr_cal(acc_all, train_len, CLASSES_NUM)

        save_list=[csp,clf]
        f=open(current_working_dir+ args.model_save_path + str(personid)+".m", "wb+")
        joblib.dump(save_list,f)
        f.close()
        print ("Done\n")


    elif args.model_type == 'Transformer':
         ###########Initial Parameters############
        c_dim = 4
        dimension = (190, 50)
        start_epoch = 0
        Tensor = torch.cuda.FloatTensor
        LongTensor = torch.cuda.LongTensor
        LR=args.lr
        EPOCH =args.epoch
        BATCH_SIZE = args.bs
        model_num=args.model_num
        #########################################
        csp_path =  current_working_dir + args.model_save_path + str(personid) +  '_csp.npy'
        tmp_train_data = np.transpose(np.squeeze(data_X_train.copy()), (0, 2, 1)) 
        if CLASSES_NUM==2:
            Wb = common_spatial_pattern.csp_two(tmp_train_data, data_Y_train)  # common spatial pattern
        elif CLASSES_NUM==4:
            Wb = common_spatial_pattern.csp_four(tmp_train_data, data_Y_train) 
        data_X_train = np.einsum('acd, ce -> aed', data_X_train, Wb)
        data_X_evl = np.einsum('acd, ce -> aed', data_X_evl, Wb)
        np.save(csp_path,Wb)
        #########################################

        print("===================start training====================")
        itr_all=[]
        acc_all=[]
        for i in range(model_num):
            print('Model===%d'%(i+1))
            seedval = args.seed + i
            seed_torch(seedval)
            
            model_name = current_working_dir + args.model_save_path + str(personid) + '_' + str(seedval) + '.pth'
            losstrain,lossval,train_acc_all,val_acc_all,Best_EPOCH,Best_ACC=mod_train(X_TRAIN=data_X_train, Y_TRAIN=data_Y_train, X_VALID=data_X_evl, Y_VALID=data_Y_evl,model_save_path=model_name)
            itr_i=itr_cal(Best_ACC, train_len, CLASSES_NUM)
            itr_all.append(itr_i)
            acc_all.append(Best_ACC)
            print("validation ACC : {}, ITR : {} , best epoch : {}".format(Best_ACC,itr_i,Best_EPOCH))



        
        
    
    print("===========================================================================================================")

    print("\n dataset: {} \n classes: {} \n {} model: {} \n augmentation: {}  \n train_len: {} \n personid: {} \n mean acc: {} \n mean itr: {} \n ".format(args.dataset,args.classes,"within",args.model_type,args.augmentation, args.train_len, personid, np.mean(acc_all),np.mean(itr_all)))      


