# -*- coding:utf-8 -*-
import datetime

import numpy as np
import torch
import torch.nn as nn
import util
from models import BCL_Network
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from dimensionality_reduction_com import logistic_dimension
import prepare as pre

# from tensorboardX import SummaryWriter
workDir = '/home/zjj/code/3/Dimensional_reduction/'
dataDir = '/home/zjj/Data/'
#dataDir = '/home/caiyideng/zjj/NewData/SmallData/'
modelDir = workDir + 'model'

# 返回每一折多个epoch中的最优模型
def train(myDataLoader, fold):
    best = 0
    for epoch in range(Epoch):
        for step, (x, y) in enumerate(myDataLoader):
            model.train()
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ROC, PR, F1, test_loss, accuracy = validate(validate_DataLoader, epoch)
        if ROC > best:
            best = ROC
            model_name = modelDir  + '/validate_params_' + str(fold) + '_' + str(now_time) + '.pkl'
            torch.save(model.state_dict(), model_name)
    scheduler.step(test_loss)
    print(model_name)
    return model_name


def validate(myDataLoader, epoch):
    output_list = []
    output_result_list = []
    correct_list = []
    test_loss = 0
    for step, (x, y) in enumerate(myDataLoader):  # 每一步 loader 释放一小批数据用来学习
        model.eval()
        output = model(x)
        loss = loss_func(output, y)
        test_loss += float(loss)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    y_pred = np.array(output_result_list)
    y_true = np.array(correct_list)
    accuracy = accuracy_score(y_true, y_pred)
    test_loss /= myDataLoader.__len__()
    print('Validate set: Average loss:{:.4f}\tAccuracy:{:.3f}'.format(test_loss, accuracy))
    ROC, PR, F1 = util.get_ROC_Curve(output_list, output_result_list, correct_list)
    print('第{}折_第{}轮_ROC:{}\tPR:{}\tF1:{} '.format(fold, epoch, ROC, PR, F1))
    return ROC, PR, F1, test_loss, accuracy

def test(myDataLoader, fold, best_model_name):
    name = 'validate_params_' + str(fold)+ '_' + str(now_time)
    model.load_state_dict(torch.load(best_model_name))
    output_list = []
    output_result_list = []
    correct_list = []
    for step, (x, y) in enumerate(myDataLoader):
        model.eval()
        output = model(x)
        output_list += output.cpu().detach().numpy().tolist()
        output = (output > 0.5).int()
        output_result_list += output.cpu().detach().numpy().tolist()
        correct_list += y.cpu().detach().numpy().tolist()
    accuracy = accuracy_score(correct_list, output_result_list)
    precision = precision_score(correct_list, output_result_list, average='macro')
    recall = recall_score(correct_list, output_result_list, average='macro')
    print(confusion_matrix(correct_list, output_result_list))
    #print(classification_report(correct_list, output_result_list))
    util.draw_ROC_Curve(output_list, output_result_list, correct_list, name)
    ROC, PR, F1 = util.draw_PR_Curve(output_list, output_result_list, correct_list, name)
    return ROC, PR, F1,accuracy,precision,recall


def getDataSet(train_index, test_index):
    x_train = np.array(X)[train_index]
    y_train = np.array(y)[train_index]
    x_test = np.array(X)[test_index]
    y_test = np.array(y)[test_index]
    x_train_, x_validate_, y_train_, y_validate_ = train_test_split(
        x_train, y_train, test_size=0.125, stratify=y_train)

    #实例化类DealDataSet，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader
    x_train_ = torch.from_numpy(x_train_).type(torch.FloatTensor).cuda()
    y_train_ = torch.Tensor(y_train_).cuda()
    x_validate_ = torch.from_numpy(x_validate_).type(torch.FloatTensor).cuda()
    y_validate_ = torch.Tensor(y_validate_).cuda()
    x_test = torch.from_numpy(x_test).type(torch.FloatTensor).cuda()
    y_test = torch.Tensor(y_test).cuda()
    train_DataSet = TensorDataset(x_train_, y_train_)
    validate_DataSet = TensorDataset(x_validate_,y_validate_)
    test_DataSet = TensorDataset(x_test,y_test)
    train_DataLoader = DataLoader(dataset=train_DataSet, batch_size=Batch_Size, shuffle=True)
    validate_DataLoader = DataLoader(dataset=validate_DataSet, batch_size=test_Batch_Size, shuffle=True)
    test_DataLoader = DataLoader(dataset=test_DataSet, batch_size=test_Batch_Size, shuffle=True)

    return train_DataLoader, validate_DataLoader, test_DataLoader


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # PyTorch v0.4.0
    #这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
    # 常用参数
    Batch_Size = 16
    test_Batch_Size = 32
    LR = 0.0001
    Epoch = 100
    K_Fold = 5
    print("Batch_Size", Batch_Size)
    print("LR",LR)
    print("Epoch", Epoch)
    print("K_Fold", K_Fold)
    now_time = datetime.datetime.now().strftime('%H:%M:%S')

    data, label = pre.CreateData(dataDir + "/label.csv", dataDir + "/replace_data.csv",dataDir + "/data.csv")
    index = [i for i in range(len(data))]
    index = np.array(index)
    data_2, mask = logistic_dimension(data,label, parameter=1)
    shu = data_2
    X = shu[index,:]
    y = label

    kf = StratifiedKFold(n_splits=K_Fold, shuffle=True)
    fold = 1
    roc_total = []
    pr_total = []
    F1_total = []
    acc_total = []
    pre_total = []
    recall_total = []

    for train_index, validate_index in kf.split(X, y):
        # writer = SummryWriter(comment='test')
        train_DataLoader, validate_DataLoader, test_DataLoader = getDataSet(train_index, validate_index)
        model = BCL_Network().to(device)
        # model = nn.parallel.DataParallel(model, device_ids=[0, 1, 2, 3])
        #  优化器和损失函数writer
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        # 动态学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
        loss_func = nn.BCELoss().to(device)
        best_model_name = train(train_DataLoader, fold)
        ROC, PR, F1, acc, precision, recall = test(test_DataLoader, fold, best_model_name)
        roc_total.append(ROC)
        pr_total.append(PR)
        F1_total.append(F1)
        acc_total.append(acc)
        pre_total.append(precision)
        recall_total.append(recall)
        fold += 1

    # 获得平均值
    roc_average = np.mean(roc_total)
    pr_average = np.mean(pr_total)
    f1_average = np.mean(F1_total)
    acc_average = np.mean(acc_total)
    pre_average = np.mean(pre_total)
    recall_average = np.mean(recall_total)
    print("Average ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
    print("Average acc:{}\tprecision:{}\trecall:{}".format(acc_average, pre_average, recall_average))
    print("#################################")
