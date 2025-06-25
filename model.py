import torch
import torch.nn as nn
import torch.nn.functional as F

class ShapeConv(nn.Module):
    def __init__(self, in_channels, out_channels, shapelet_length, num_classes=None, supervised=True):
        """
        初始化ShapeConv层

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数（即shapelet数量）
            shapelet_length (int): 每个shapelet的长度
            num_classes (int): 分类任务的类别数（监督学习时使用）
            supervised (bool): 是否为监督学习
        """
        super(ShapeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapelet_length = shapelet_length
        self.num_classes = num_classes
        self.supervised = supervised

        # 定义卷积核（shapelets）
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=shapelet_length,
            stride=1,
            padding=0,
            bias=False
        )

        # 初始化卷积核
        self._initialize_shapelets()

    def _initialize_shapelets(self):
        """
        初始化shapelet权重（监督或无监督）
        """
        with torch.no_grad():
            if self.supervised and self.num_classes is not None:
                # 监督学习初始化：按类别分配shapelet
                k = self.out_channels // self.num_classes
                for i in range(self.num_classes):
                    for j in range(k):
                        # 假设数据按时间轴分割，初始化为数据子序列均值
                        # 这里简化为随机初始化，实际需从数据中采样
                        torch.manual_seed(42 + i * k + j)  # 使用一个固定的、可预测的种子, 这样可以保证每次运行，第n个shapelet的初始值都一样, 但又与其他shapelet不同
                        self.conv.weight[i * k + j] = torch.randn(self.in_channels, self.shapelet_length)
            else:
                # 无监督学习初始化：基于KMeans预聚类
                # 这里简化为随机初始化，实际需实现KMeans聚类
                self.conv.weight.data = torch.randn(self.out_channels, self.in_channels, self.shapelet_length)

    # def shape_regularization(self, x):
    #     """
    #     计算shape正则化项，保持shapelet与数据子序列的相似性
    #
    #     参数:
    #         x (torch.Tensor): 输入时间序列，形状为 (batch_size, in_channels, seq_length)
    #
    #     返回:
    #         torch.Tensor: shape正则化损失
    #     """
    #     batch_size, _, seq_length = x.shape
    #     reg_loss = 0.0
    #
    #     # 对每个shapelet计算与输入子序列的最小距离
    #     for i in range(self.out_channels):
    #         shapelet = self.conv.weight[i]  # (in_channels, shapelet_length)
    #         min_dist = float('inf')
    #
    #         # 滑动窗口提取子序列
    #         for j in range(seq_length - self.shapelet_length + 1):
    #             sub_seq = x[:, :, j:j + self.shapelet_length]  # (batch_size, in_channels, shapelet_length)
    #             dist = torch.mean((shapelet - sub_seq) ** 2, dim=(1, 2))  # 平方欧几里得距离
    #             min_dist = min(min_dist, dist.mean().item())
    #
    #         reg_loss += min_dist
    #
    #     return reg_loss / self.out_channels

    # 在 model.py -> class ShapeConv 中
    # 用这个新函数替换掉旧的 shape_regularization 函数

    def shape_regularization(self, x):
        """
        以向量化且可微分的方式计算 shape 正则化损失。
        """
        # 这个计算过程与 forward 函数中的距离计算相呼应，
        # 但我们确保得到的是真实的平方欧几里得距离。
        conv_out = self.conv(x)

        # 计算范数的平方
        shapelet_norm_sq = torch.sum(self.conv.weight ** 2, dim=(1, 2))

        sub_sequences = x.unfold(dimension=2, size=self.shapelet_length, step=1)
        sub_sequences_norm_sq = torch.sum(sub_sequences ** 2, dim=(1, 3))

        # 通过广播（broadcasting）来计算配对的范数和
        # shapelet_norm_sq: (out_channels) -> (1, out_channels, 1)
        # sub_sequences_norm_sq: (batch_size, seq_len') -> (batch_size, 1, seq_len')
        norm_term = shapelet_norm_sq.view(1, -1, 1) + sub_sequences_norm_sq.unsqueeze(1)

        # 平方欧几里得距离: D^2 = S^2 + T^2 - 2*S*T
        # shapelet_distances 的形状为 (batch_size, out_channels, seq_len')
        shapelet_distances = norm_term - 2 * conv_out

        # 对每个 shapelet，在每个样本中找到其与所有子序列的最小距离
        # min() 返回一个元组 (values, indices)，我们只需要值。
        min_distances, _ = torch.min(shapelet_distances, dim=2)  # 形状变为 (batch_size, out_channels)

        # 正则化损失是这些最小距离的平均值
        # 在所有 shapelet 和批次中的所有样本上取平均。
        reg_loss = torch.mean(min_distances)

        return reg_loss


    def diversity_regularization(self):
        """
        计算多样性正则化项，防止shapelet过于相似

        返回:
            torch.Tensor: 多样性正则化损失
        """
        weights = self.conv.weight  # (out_channels, in_channels, shapelet_length)
        dist_matrix = torch.zeros(self.out_channels, self.out_channels)

        for i in range(self.out_channels):
            for j in range(self.out_channels):
                if i != j:
                    dist = torch.exp(-torch.norm(weights[i] - weights[j], p=2))
                    dist_matrix[i, j] = dist

        return torch.norm(dist_matrix, p='fro')

    def forward(self, x):
        """
        前向传播，计算shapelet变换

        参数:
            x (torch.Tensor): 输入时间序列，形状为 (batch_size, in_channels, seq_length)

        返回:
            torch.Tensor: shapelet变换后的特征
        """
        conv_out = self.conv(x)
        shapelet_norm = torch.sum(self.conv.weight ** 2, dim=(1, 2)) / 2
        batch_size, _, seq_length = x.shape
        sub_sequences = x.unfold(dimension=2, size=self.shapelet_length,
                                 step=1)  # (batch_size, in_channels, seq_length - shapelet_length + 1, shapelet_length)
        input_norm = torch.sum(sub_sequences ** 2, dim=(1, 3)) / 2  # (batch_size, seq_length - shapelet_length + 1)

        # 调整形状以进行广播
        shapelet_norm = shapelet_norm.view(1, -1, 1)  # (1, out_channels, 1)
        input_norm = input_norm.view(batch_size, 1, -1)  # (batch_size, 1, seq_length - shapelet_length + 1)
        norm_term = shapelet_norm + input_norm  # 广播后形状为 (batch_size, out_channels, seq_length - shapelet_length + 1)

        # shapelet 变换：卷积结果 - 范数项
        features = conv_out - norm_term  # 计算 Y_ij - N

        # 最大池化
        features = F.max_pool1d(features, kernel_size=features.size(-1)).squeeze(-1)

        features = -2 * features

        return features


class ShapeConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, shapelet_length, num_classes, supervised=True):
        """
        完整的ShapeConv模型，包括ShapeConv层和分类器

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数（shapelet数量）
            shapelet_length (int): 每个shapelet的长度
            num_classes (int): 分类任务的类别数
            supervised (bool): 是否为监督学习
        """
        super(ShapeConvModel, self).__init__()
        self.shapeconv = ShapeConv(in_channels, out_channels, shapelet_length, num_classes, supervised)
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.supervised = supervised

    def forward(self, x):
        """
        模型前向传播

        参数:
            x (torch.Tensor): 输入时间序列，形状为 (batch_size, in_channels, seq_length)

        返回:
            torch.Tensor: 分类输出
        """
        features = self.shapeconv(x)
        return self.classifier(features)

    def compute_loss(self, x, y=None, lambda_shape=0.1, lambda_div=0.1):
        """
        计算总损失，包括任务损失、shape正则化和多样性正则化

        参数:
            x (torch.Tensor): 输入时间序列
            y (torch.Tensor): 标签（监督学习时使用）
            lambda_shape (float): shape正则化权重
            lambda_div (float): 多样性正则化权重

        返回:
            torch.Tensor: 总损失
        """
        features = self.shapeconv(x)
        logits = self.classifier(features)
        task_loss = F.cross_entropy(logits, y)
        shape_reg = self.shapeconv.shape_regularization(x)
        div_reg = self.shapeconv.diversity_regularization()

        return task_loss + lambda_shape * shape_reg + lambda_div * div_reg