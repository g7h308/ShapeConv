import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def mean_std(train_data):
    """Compute mean and std for normalization as in code segment 1."""
    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)
    s_len = np.std(train_data, axis=2)
    std = np.max(s_len, axis=0)
    return mean, std


def mean_std_transform(data, mean, std):
    """Apply mean-std normalization as in code segment 1."""
    return (data - mean) / std


def load(config):
    Data = {}
    fold_num = config['fold_num']
    problem = config['data_dir'].split('/')[-1]
    npy_path = os.path.join(config['data_dir'], f"{problem}_fold{fold_num}.npy")

    # 检查是否已存在预处理数据
    if os.path.exists(npy_path):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(npy_path, allow_pickle=True)
        Data['max_len'] = Data_npy.item().get('max_len')
        Data['X_train'] = Data_npy.item().get('X_train')
        #Data['X_train'] = Data['X_train'].transpose(0, 2, 1)
        Data['y_train'] = Data_npy.item().get('y_train')
        Data['X_val'] = Data_npy.item().get('X_val')
        #Data['X_val'] = Data['X_val'].transpose(0, 2, 1)
        Data['y_val'] = Data_npy.item().get('y_val')
        Data['X_test'] = Data_npy.item().get('X_test')
        #Data['X_test'] = Data['X_test'].transpose(0, 2, 1)
        Data['y_test'] = Data_npy.item().get('y_test')

        logger.info(f"{len(Data['y_train'])} samples will be used for training")
        logger.info(f"{len(Data['y_val'])} samples will be used for validation")
        logger.info(f"{len(Data['y_test'])} samples will be used for testing")

    else:
        logger.info("Loading and preprocessing data ...")
        # 扩大倍数   任务态VFT  n=2  静息态PREP n=4
        n = 4
        adhd_dir = os.path.join(config['data_dir'], "ADHD")
        hc_dir = os.path.join(config['data_dir'], "HC")

        # 读取所有Excel文件
        adhd_files = [os.path.join(adhd_dir, f) for f in os.listdir(adhd_dir) if f.endswith('.xlsx')]
        hc_files = [os.path.join(hc_dir, f) for f in os.listdir(hc_dir) if f.endswith('.xlsx')]

        # 加载数据并分配标签
        dataframes = []
        labels = []
        group_ids = []  # NEW: 用于跟踪每个样本的组标识

        # 处理 ADHD 数据 (label = 1)
        for idx, file in enumerate(adhd_files):
            df = pd.read_excel(file)
            dataframes.append(df)
            labels.append(1)  # ADHD标签为1
            group_ids.append(idx)  # NEW: 为每个 ADHD 文件分配唯一 group_id

        # 处理 HC 数据 (label = 0)
        for idx, file in enumerate(hc_files, len(adhd_files)):  # NEW: 从 len(adhd_files) 开始继续编号
            df = pd.read_excel(file)
            dataframes.append(df)
            labels.append(0)  # HC标签为0
            group_ids.append(idx)  # NEW: 为每个 HC 文件分配唯一 group_id

        # 转换为numpy数组
        feature = np.array(dataframes)
        labels = np.array(labels)
        group_ids = np.array(group_ids)  # NEW: 转换为 numpy 数组
        logger.info(f"feature 原始shape：{feature.shape}")

        # 获取原始数据的形状
        n_samples, n_time, n_features = feature.shape

        # 计算可被n整除的最大时间长度
        n_time_adjusted = (n_time // n) * n
        logger.info(f"调整后的时间维度长度: {n_time_adjusted}（原始: {n_time}）")

        # 数据扩增：将时间维度分割为n份
        split_features = []
        split_labels = []
        split_groups = []  # NEW: 用于存储扩展后的 group_ids
        for i in range(n):
            indices = np.arange(i, n_time_adjusted, n)
            split_data = feature[:, indices, :]
            split_features.append(split_data)
            split_labels.append(labels)
            split_groups.append(group_ids)  # NEW: 扩展 group_ids，保持与样本对应

        # 合并分割后的特征和标签
        feature = np.concatenate(split_features, axis=0)  # 形状变为 (n*n_samples, n_time//n, n_features)
        label = np.concatenate(split_labels, axis=0)  # 形状变为 (n*n_samples,)
        groups = np.concatenate(split_groups, axis=0)  # NEW: 形状变为 (n*n_samples,)
        logger.info(f"feature 合并后shape：{feature.shape}")

        # 数据标准化（使用代码段1的mean_std和mean_std_transform）

        mean, std = mean_std(feature)
        mean = np.repeat(mean, feature.shape[1]).reshape(feature.shape[2], feature.shape[1])
        std = np.repeat(std, feature.shape[1]).reshape(feature.shape[2], feature.shape[1])
        feature = mean_std_transform(feature, mean, std)

        # 设置max_len
        Data['max_len'] = feature.shape[1]  # 时间维度长度

        feature = feature.transpose(0, 2, 1)

        # 先将数据分为 train+val（80%）和 test（20%）
        gss = GroupShuffleSplit(test_size=0.2, random_state=42)
        train_val_idx, test_idx = next(gss.split(feature, label, groups))
        X_train_val = feature[train_val_idx]
        y_train_val = label[train_val_idx]
        groups_train_val = groups[train_val_idx]
        X_test = feature[test_idx]
        y_test = label[test_idx]

        # 对 train+val 进行 5 折交叉验证
        gkf = GroupKFold(n_splits=5)
        fold_indices = list(gkf.split(X_train_val, y_train_val, groups_train_val))

        # NEW: 验证 fold_num 是否有效
        if fold_num is None or fold_num not in range(5):
            raise ValueError("fold_num 必须为 0 到 4 的整数")

        # 选择指定折的训练和验证索引
        train_idx, val_idx = fold_indices[fold_num]
        X_train = X_train_val[train_idx]
        y_train = y_train_val[train_idx]
        X_val = X_train_val[val_idx]
        y_val = y_train_val[val_idx]

        # 打印当前折的结果
        print(f"\n=== 第 {fold_num} 折 ===")
        print(f"训练集大小: {X_train.shape[0]} 样本, 组: {np.unique(groups_train_val[train_idx])}")
        print(f"验证集大小: {X_val.shape[0]} 样本, 组: {np.unique(groups_train_val[val_idx])}")

        Data['X_train'] = X_train
        Data['y_train'] = y_train
        Data['X_val'] = X_val
        Data['y_val'] = y_val
        Data['X_test'] = X_test
        Data['y_test'] = y_test



        # 合并训练集和验证集的特征
        Data['All_train_data'] = np.concatenate([Data['X_train'], Data['X_val']], axis=0)

        # 合并训练集和验证集的标签
        Data['All_train_label'] = np.concatenate([Data['y_train'], Data['y_val']], axis=0)

        logger.info(f"{len(y_train)} samples will be used for training")
        logger.info(f"{len(y_val)} samples will be used for validation")
        #logger.info(f"{len(labels[test_idx])} samples will be used for testing")

        # 保存预处理数据
        np.save(npy_path, Data, allow_pickle=True)

    return Data