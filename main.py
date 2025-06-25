import argparse
import logging

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from model import ShapeConvModel
import dataloader
import copy
import time
from torch.utils.data import DataLoader, TensorDataset

from visualize import visualize_shapelets_with_location

#配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--fold_num', default=0, type=int)
parser.add_argument('--data_path', default='../TSCModel/RankSCL/RankSCL/ADHD')
parser.add_argument('--problem', default='VFT')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--shapelets_num', default=6, type=int, help='总的shapelets数量，每个class均分')
parser.add_argument('--ratio', default=0.3, type=float, help= 'shaplets长度占时间序列长度的比例')
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--lambda_shape',default=1e-2)
parser.add_argument('--lambda_div',default=1e-2)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#模型训练
def train_model_process(model,train_dataloader,val_dataloader,config):

    print("Using device: ", device)

    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=config['lr'])

    #损失函数
    criterion = nn.CrossEntropyLoss()

    #模型放入device
    model = model.to(device)

    #保存最好的模型，这里初始化为初始model
    best_model_wts = copy.deepcopy(model.state_dict())

    #最高精度，初始设为0.0
    best_acc = 0.0

    #训练、验证损失列表，保存每一次训练和验证的loss值
    train_loss_list , val_loss_list = [],[]
    #训练、验证精度列表，保存每一次训练和验证的精度
    train_acc_list, val_acc_list = [], []

    #记录当前时间，用于计算每一轮的消耗的时间
    since=time.time()
    num_epochs = config['epochs']
    for epoch in range(num_epochs):
        print("Epoch{}/{}".format(epoch+1,num_epochs))
        print("\n")
        train_loss = 0.0
        train_acc = 0


        val_loss = 0.0
        val_acc = 0

        #训练集和验证集的样本数量
        train_num = 0
        val_num = 0
        # 打开训练模式
        model.train()
        #训练
        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            # print(b_y)
            pre_lab = torch.argmax(output,dim=1)

            loss = criterion(output,b_y)


            shape_reg = model.shapeconv.shape_regularization(b_x)
            div_reg = model.shapeconv.diversity_regularization()

            total_loss = loss + config['lambda_shape'] * shape_reg + config['lambda_div'] * div_reg

            optimizer.zero_grad()

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            #对损失函数进行累加
            train_loss +=total_loss.item()*b_x.size(0)
            #对精确度累加
            train_acc += torch.sum(pre_lab == b_y.data)
            #当前已经被用于训练的样本数量累加
            train_num += b_x.size(0)

        #验证
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            #打开验证模式
            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            shape_reg = model.shapeconv.shape_regularization(b_x)
            div_reg = model.shapeconv.diversity_regularization()

            total_loss = loss + config['lambda_shape'] * shape_reg + config['lambda_div'] * div_reg

            # 对损失函数进行累加
            val_loss += total_loss.item() * b_x.size(0)
            # 对精确度累加
            val_acc += torch.sum(pre_lab == b_y.data)
            # 当前已经被用于训练的样本数量累加
            val_num += b_x.size(0)


        train_loss_list.append(train_loss / train_num)
        train_acc_list.append(train_acc.double().item() / train_num)

        val_loss_list.append(val_loss / val_num)
        val_acc_list.append(val_acc.double().item() / val_num)

        print("第{}轮  trainloss：{:.4f}  train acc：{:.4f}".format(epoch+1,train_loss_list[-1],train_acc_list[-1]))
        print("第{}轮  valloss：  {:.4f}  val acc：  {:.4f}".format(epoch+1,val_loss_list[-1],val_acc_list[-1]))

        time_use = time.time()-since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))
        print("-"*20)
        print("\n")
        #保存最高精确度模型
        if val_acc_list[-1] > best_acc:
            best_acc = val_acc_list[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts,'./best_model.pth')


def test_model_process(model, test_dataloader):
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            test_data_x = test_data_x.to(device)
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            output= model(test_data_x)
            pre_lab = torch.argmax(output, dim=1)
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)



# 示例用法
if __name__ == "__main__":
    config = args.__dict__
    config['data_dir'] = config['data_path'] + "/" + config['problem']
    data = dataloader.load(config)
    print(data['X_train'].shape)
    print(data['y_train'].shape)


    # 输入参数

    in_channels = len(data['X_train'][0])
    seq_length = data['max_len']
    out_channels = config['shapelets_num']
    shapelet_length = int(seq_length*config['ratio'])
    num_classes = np.max(data['y_train'])+1



    # 创建模型
    model = ShapeConvModel(
        in_channels=in_channels,
        out_channels=out_channels,
        shapelet_length=shapelet_length,
        num_classes=num_classes,
        supervised=True
    )

    x_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train_np = data['y_train']  # 获取numpy格式的y_train用于可视化
    y_train = torch.tensor(data['y_train'], dtype=torch.long)
    x_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.long)
    x_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.long)


    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test,y_test)

    # Define batch size (adjust as needed)
    batch_size = config['batch_size']

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size = batch_size,shuffle=False)

    #训练前可视化
    visualize_shapelets_with_location(model,x_train, all_labels=y_train_np,title_prefix="Before Training", shapelet_length=shapelet_length, save_dir="visualizations")

    # 训练模型
    train_model_process(model,train_loader,val_loader,config)

    #训练后可视化与测试
    logging.info("Loading best model for final visualization and testing...")
    model.load_state_dict(torch.load('best_model.pth'))
    visualize_shapelets_with_location(model, x_train, all_labels=y_train_np,title_prefix="After Training", shapelet_length=shapelet_length, save_dir="visualizations")

    test_model_process(model, test_loader)