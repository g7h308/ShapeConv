# visualize.py (保存为文件版)

import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

# 在这个版本中，我们不需要关心后端，所以可以移除相关代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_shapelets_with_location(model, dataset, all_labels, title_prefix="", shapelet_length=None,
                                      save_dir="visualizations"):
    """
    可视化Shapelets，并将一个足够长的静态图片保存到文件。

    参数:
        model (torch.nn.Module): 训练好的或初始化的模型。
        dataset (torch.Tensor): 用于可视化的数据集 (e.g., x_train)，形状 (N, C, L)。
        all_labels (np.array): 数据集对应的标签，形状 (N,)。
        title_prefix (str): 图表标题和文件名的前缀。
        shapelet_length (int): Shapelet的长度。
        save_dir (str): 保存图片的目录。
    """
    logging.info(f"Generating and saving visualization: {title_prefix}")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    model.to(device)
    dataset = dataset.to(device)

    unique_classes = np.unique(all_labels)

    # 1. 获取模型和数据信息 (代码不变)
    try:
        shapelets = model.shapeconv.conv.weight.detach()
    except AttributeError:
        logging.error("Model visualization failed: model does not have 'shapeconv.conv.weight'.")
        return
    in_channels, out_channels = shapelets.shape[1], shapelets.shape[0]
    if shapelet_length is None: shapelet_length = shapelets.shape[2]

    # 2. 计算部分 (代码不变)
    with torch.no_grad():
        conv_output = model.shapeconv.conv(dataset)
        predicted_classes = []
        for s_idx in range(out_channels):
            responses_for_s = conv_output[:, s_idx, :]
            max_responses_per_sample, _ = torch.max(responses_for_s, dim=1)
            avg_response_per_class = []
            for c in unique_classes:
                class_mask = (torch.tensor(all_labels) == c)
                if class_mask.any():
                    avg_response_per_class.append(max_responses_per_sample[class_mask].mean().item())
                else:
                    avg_response_per_class.append(-float('inf'))
            predicted_classes.append(unique_classes[np.argmax(avg_response_per_class)])
        predicted_classes = np.array(predicted_classes)
        shapelet_norms_per_channel = torch.norm(shapelets, p=2, dim=2)
        dominant_channels = torch.argmax(shapelet_norms_per_channel, dim=1).cpu().numpy()
        conv_output_flat = conv_output.permute(1, 0, 2).reshape(out_channels, -1)
        _, flat_indices = torch.max(conv_output_flat, dim=1)
        num_timesteps_out = conv_output.shape[2]
        best_sample_indices = (flat_indices // num_timesteps_out).cpu().numpy()
        best_time_indices = (flat_indices % num_timesteps_out).cpu().numpy()

    # 打印信息 (代码不变)
    # ...

    # 3. 绘图部分 (核心修改在这里)
    data_np = dataset.cpu().numpy()
    shapelets_np = shapelets.cpu().numpy()

    # 动态计算一个非常高的高度，确保每个子图都不拥挤。
    # 每个通道分配约 3 英寸的高度，宽度固定。
    fig_height = max(8, in_channels * 3)
    fig, axes = plt.subplots(in_channels, 1, figsize=(20, fig_height), sharex=True, squeeze=False)
    fig.suptitle(f'{title_prefix} - Shapelets by Dominant Channel & Predicted Class', fontsize=20)

    for i in range(in_channels):
        ax = axes[i, 0]
        # 绘制背景数据 (根据规则1修改)
        # 规则1: 0 -> 浅蓝色, 1 -> 浅橙色
        background_colors = {0: 'lightskyblue', 1: 'moccasin'}
        for sample_idx, sample_data in enumerate(data_np):
            label = all_labels[sample_idx]
            color = background_colors.get(label, 'lightgray')  # 如果有未知的标签，默认为灰色
            ax.plot(sample_data[i, :], color=color, alpha=0.4, zorder=1)

        # 绘制 shapelets (根据规则2修改)
        shapelets_on_this_channel_indices = np.where(dominant_channels == i)[0]
        for c in unique_classes:
            class_specific_shapelets = [s_idx for s_idx in shapelets_on_this_channel_indices if
                                        predicted_classes[s_idx] == c]
            if not class_specific_shapelets: continue

            # 规则2: 0 -> 深蓝色 (使用'Blues'色带), 1 -> 深红色 (使用'Reds'色带)
            cmap_name = 'Reds' if c == 1 else 'Blues'
            # 使用更深的颜色范围，从0.95(非常深)到0.6(较深)，以满足 "深色" 的要求并区分shapelet
            colors = plt.cm.get_cmap(cmap_name)(np.linspace(0.95, 0.6, len(class_specific_shapelets)))

            for color_idx, s_idx in enumerate(class_specific_shapelets):
                start = best_time_indices[s_idx]
                x_range = np.arange(start, start + shapelet_length)
                shapelet_component = shapelets_np[s_idx, i, :]
                # 明确标签是shapelet的"预测类别"
                ax.plot(x_range, shapelet_component, color=colors[color_idx], linewidth=2.5,
                        label=f'Shapelet {s_idx + 1} (Pred. Class {c})', zorder=2)

        # 设置标题和图例 (代码不变)
        class_counts = {c: np.sum((dominant_channels == i) & (predicted_classes == c)) for c in unique_classes}
        title_str = f'Channel {i} | ' + ' | '.join(
            [f'Class {c} Shapelets: {count}' for c, count in class_counts.items()])
        ax.set_title(title_str, fontsize=14)
        if len(shapelets_on_this_channel_indices) > 0:
            ax.legend(loc='best', fontsize='medium')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel("Time Step", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # 4. 保存文件并关闭图形对象
    # 将文件名中的空格替换为下划线
    safe_title = title_prefix.replace(" ", "_")
    save_path = os.path.join(save_dir, f"{safe_title}_visualization.png")

    # 使用高DPI保存，以获得更清晰的图片
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f"Visualization saved to: {save_path}")

    # 关闭图形，防止在内存中累积
    plt.close(fig)