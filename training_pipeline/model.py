import torch
import pytorch_lightning as pl
from torch import nn, optim, Tensor
from dataclasses import asdict
from typing import Callable, List
from training_pipeline.metric_calculators import (
    MetricCalculator,
)
from training_pipeline.metrics_containers import (
    MetricContainer,
)


class BottleneckBlock(nn.Module):
    """
    倒置瓶颈层。
    来源于论文 "Scaling MLPs: A Tale of Inductive Bias" https://arxiv.org/pdf/2306.13575.pdf。
    该层的主要思想是首先将输入扩展到更宽的隐藏维度，然后应用非线性激活函数，
    最后将结果投影回原始维度。这种结构可以在保持模型参数量较小的同时提高模型的表达能力。
    """

    def __init__(self, thin_dim: int, wide_dim: int):
        """
        初始化倒置瓶颈层。

        参数:
            thin_dim (int): 输入和输出的维度
            wide_dim (int): 中间隐藏层的维度，通常大于thin_dim
        """
        super().__init__()
        self.l1 = nn.Linear(thin_dim, wide_dim)
        self.l2 = nn.Linear(wide_dim, thin_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
            x (torch.Tensor): 输入张量

        返回:
            torch.Tensor: 经过瓶颈层处理后的输出张量
        """
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


class Net(nn.Module):
    """
    神经网络主体结构。
    该网络使用多个倒置瓶颈层和层归一化来处理输入数据，
    通过残差连接提高网络的训练稳定性和性能。
    """

    def __init__(
        self,
        embedding_dim: int,
        output_dim: int,
        hidden_size_wide: int,
        hidden_size_thin: int,
    ):
        """
        初始化神经网络。

        参数:
            embedding_dim (int): 输入嵌入维度
            output_dim (int): 输出维度
            hidden_size_wide (int): 瓶颈层中的宽维度
            hidden_size_thin (int): 瓶颈层中的窄维度
        """
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, hidden_size_thin)
        self.ln_input = nn.LayerNorm(normalized_shape=hidden_size_thin)

        self.layernorms = nn.ModuleList(
            [nn.LayerNorm(normalized_shape=hidden_size_thin) for _ in range(3)]
        )
        self.bottlenecks = nn.ModuleList(
            [
                BottleneckBlock(thin_dim=hidden_size_thin, wide_dim=hidden_size_wide)
                for _ in range(3)
            ]
        )

        self.ln_output = nn.LayerNorm(normalized_shape=hidden_size_thin)
        self.linear_output = nn.Linear(hidden_size_thin, out_features=output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数。

        参数:
            x (Tensor): 输入张量

        返回:
            Tensor: 模型的输出预测
        """
        x = self.input_projection(x)
        x = self.ln_input(x)
        for layernorm, bottleneck in zip(self.layernorms, self.bottlenecks):
            x = x + bottleneck(layernorm(x))
        x = self.ln_output(x)
        x = self.linear_output(x)
        return x


class UniversalModel(pl.LightningModule):
    """
    通用推荐模型。
    这是一个基于PyTorch Lightning的模型类，用于训练和评估推荐系统。
    该模型整合了自定义的神经网络结构、损失函数和评估指标。
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_size_thin: int,
        hidden_size_wide: int,
        output_dim: int,
        learning_rate: float,
        metric_calculator: MetricCalculator,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        metrics_tracker: List[MetricContainer],
    ) -> None:
        """
        初始化通用推荐模型。

        参数:
            embedding_dim (int): 输入嵌入维度
            hidden_size_thin (int): 瓶颈层中的窄维度
            hidden_size_wide (int): 瓶颈层中的宽维度
            output_dim (int): 输出维度
            learning_rate (float): 学习率
            metric_calculator (MetricCalculator): 指标计算器
            loss_fn (Callable[[Tensor, Tensor], Tensor]): 损失函数
            metrics_tracker (List[MetricContainer]): 指标跟踪器列表
        """
        super().__init__()

        torch.manual_seed(1278)
        self.learning_rate = learning_rate
        self.net = Net(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_size_thin=hidden_size_thin,
            hidden_size_wide=hidden_size_wide,
        )
        self.metric_calculator = metric_calculator
        self.loss_fn = loss_fn
        self.metrics_tracker = metrics_tracker

    def forward(self, x) -> Tensor:
        """
        模型的前向传播。

        参数:
            x: 输入数据

        返回:
            Tensor: 模型预测结果
        """
        return self.net(x)

    def configure_optimizers(self) -> optim.Optimizer:
        """
        配置优化器。

        返回:
            optim.Optimizer: AdamW优化器实例
        """
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def setup(self, stage):
        """
        设置阶段的初始化操作。

        参数:
            stage: 训练阶段标识
        """
        self.metric_calculator.to(self.device)

    def training_step(self, train_batch, batch_idx) -> Tensor:
        """
        定义训练步骤。

        参数:
            train_batch: 训练数据批次
            batch_idx: 批次索引

        返回:
            Tensor: 计算得到的损失值
        """
        x, y = train_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx) -> None:
        """
        定义验证步骤。

        参数:
            val_batch: 验证数据批次
            batch_idx: 批次索引
        """
        x, y = val_batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        self.metric_calculator.update(
            predictions=preds,
            targets=y.long(),
        )

    def on_validation_epoch_end(self) -> None:
        """
        验证轮次结束时的操作。
        计算并记录各项评估指标，更新指标跟踪器。
        """
        metric_container = self.metric_calculator.compute()

        for metric_name, metric_val in asdict(metric_container).items():
            self.log(
                metric_name,
                metric_val,
                prog_bar=True,
                logger=True,
            )

        self.metrics_tracker.append(metric_container)
