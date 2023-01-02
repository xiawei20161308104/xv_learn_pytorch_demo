'''
Author: xv rg16xw@163.com
Date: 2022-12-21 18:46:34
LastEditors: xv rg16xw@163.com
LastEditTime: 2022-12-21 18:46:54
FilePath: \xv_learn_pytorch_demo\linear_regression.py
'''
# author : 'nickchen121';
# date: 14/4/2021 20:11

import numpy as np

# torch里要求数据类型必须是float
x = np.arange(1, 12, dtype=np.float32).reshape(-1, 1)
y = 2 * x + 3

import torch
import torch.nn as nn


# 继承nn.module，实现前向传播，线性回归直接可以看做是全连接层
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()  # 继承父类方法
        self.linear = nn.Linear(input_dim, output_dim)  # 定义全连接层，其中input_dim和output_dim是输入和输出数据的维数

    # 定义前向传播算法
    def forward(self, inp):
        out = self.linear(inp)  # 输入x后，通过全连接层得到输入出结果out
        return out  # 返回被全连接层处理后的结果


# 定义线性回归模型
regression_model = LinearRegressionModel(1, 1)  # x和y都是一维的

# 可以通过to()或者cuda()使用GPU进行模型的训练，需要将模型和数据都转换到GPU上，也可以指定具体的GPU，如.cuda(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
regression_model.to(device)

epochs = 1000  # 训练次数
learning_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(regression_model.parameters(), learning_rate)  # 优化器，这里使用随机梯度下降算法（SGD）
criterion = nn.MSELoss()  # 使用均方误差定义损失函数

for epoch in range(epochs):
    # 数据类型转换
    inputs = torch.from_numpy(x).to(device)  # 由于x是ndarray数组，需要转换成tensor类型，如果用gpu训练，则会通过to函数把数据传入gpu
    labels = torch.from_numpy(y).to(device)

    # 训练
    optimizer.zero_grad()  # 每次求偏导都会清零，否则会进行叠加
    outputs = regression_model(inputs)  # 把输入传入定义的线性回归模型中，进行前向传播
    loss = criterion(outputs, labels)  # 通过均方误差评估预测误差
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重参数

    # 每50次循环打印一次结果
    if epoch % 50 == 0:
        print("epoch:", epoch, "loss:", loss.item())

predict = regression_model(torch.from_numpy(x).requires_grad_()).data.numpy()  # 通过训练好的模型预测结果

# torch.save(regression_model.state_dict(), "model.pk1")  # 保存模型
# result = regression_model.load_state_dict(torch.load("model.pk1"))  # 加载模型