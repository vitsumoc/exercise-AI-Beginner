import pylab
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import pickle
import os
import gzip

# pick the seed for reproducability - change it to explore the effects of random variations
np.random.seed(1)
import random

# 此示例要解决的问题是根据肿瘤的大小和年龄进行恶性或良性的分类

# 随机生成分类数据
# 数据数量
n = 50
# 采样50 特征2 冗余数据0 信息数据2 噪声比例0
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0)
# X是数据样本 五十个二维向量
# Y是数据答案 这里将0/1数组改为-1/+1数组
Y = Y*2-1

# X 为float32数组 Y 为int数组
X = X.astype(np.float32); Y = Y.astype(np.int32) # features - float, label - int

# 将数据分为训练数据和测试数据 40 个训练数据 10 个测试数据
train_x, test_x = np.split(X, [ n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])

# 绘制数据集图像的方法
def plot_dataset(suptitle, features, labels):
    # 准备plot
    fig, ax = pylab.subplots(1, 1)
    # 标题
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    # 红色表示+1结果, 蓝色表示-1结果
    colors = ['r' if l>0 else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    fig.show()

# 绘制训练数据
# plot_dataset('Training data', train_x, train_labels)

# 训练示例, bias被设置为常量1
pos_examples = np.array([ [t[0], t[1], 1] for i,t in enumerate(train_x) 
                          if train_labels[i]>0])
neg_examples = np.array([ [t[0], t[1], 1] for i,t in enumerate(train_x) 
                          if train_labels[i]<0])

# 训练方法 +1示例 -1示例 训练次数
def train(positive_examples, negative_examples, num_iterations = 10000):
    
    # shape表示数组形状, 这里的值应该是 [pos数据长度, 3]
    # 因此 num_dims 是3
    num_dims = positive_examples.shape[1]
    
    # 简单把权重初始化为0 (但通常随机数好一点)
    # 此时的weights [[0.], [0.], [0.]]
    weights = np.zeros((num_dims,1)) 
    
    # +1数组长度和 -1 数组长度
    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]
    
    report_frequency = 1000
    
    # 重复训练
    for i in range(num_iterations):
        # 选择一个+1和一个-1数据
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        # 使用一个+1数据和一个-1数据进行训练, 如果结果错误, 使用选择的点来修正weights的形状
        z = np.dot(pos, weights)   
        if z < 0: # positive example was classified as negative
            weights = weights + pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0: # negative example was classified as positive
            weights = weights - neg.reshape(weights.shape)
            
        # 周期性汇报结果
        if i % report_frequency == 0:
            # 对所有 +1 结果进行计算
            pos_out = np.dot(positive_examples, weights)
            # 对所有 -1 结果进行计算
            neg_out = np.dot(negative_examples, weights)        
            # +1 的正确率
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            # -1 的正确率
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print("Iteration={}, pos correct={}, neg correct={}".format(i,pos_correct,neg_correct))

    return weights

# # 执行训练
# wts = train(pos_examples, neg_examples)
# # 显示权重
# print(wts.transpose())

# 绘制边界线
def plot_boundary(positive_examples, negative_examples, weights):
    if np.isclose(weights[1], 0):
        if np.isclose(weights[0], 0):
            x = y = np.array([-6, 6], dtype = 'float32')
        else:
            y = np.array([-6, 6], dtype='float32')
            x = -(weights[1] * y + weights[2])/weights[0]
    else:
        x = np.array([-6, 6], dtype='float32')
        y = -(weights[0] * x + weights[2])/weights[1]

    pylab.xlim(-6, 6)
    pylab.ylim(-6, 6)                      
    pylab.plot(positive_examples[:,0], positive_examples[:,1], 'bo')
    pylab.plot(negative_examples[:,0], negative_examples[:,1], 'ro')
    pylab.plot(x, y, 'g', linewidth=2.0)
    pylab.show()

# 绘制边界
# plot_boundary(pos_examples,neg_examples,wts)

# 使用测试数据测试结果
def accuracy(weights, test_x, test_labels):
    res = np.dot(np.c_[test_x,np.ones(len(test_x))],weights)
    return (res.reshape(test_labels.shape)*test_labels>=0).sum()/float(len(test_labels))

# res = accuracy(wts, test_x, test_labels)
# print(res)

# 可视化训练过程
def train_graph(positive_examples, negative_examples, num_iterations = 100):
    num_dims = positive_examples.shape[1]
    weights = np.zeros((num_dims,1)) # initialize weights
    
    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]
    
    report_frequency = 10
    snapshots = []
    
    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        # 所以其实具体怎么判断在dot里
        z = np.dot(pos, weights)   
        if z < 0:
            # 具体怎么修改在reshape里
            weights = weights + pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0:
            weights = weights - neg.reshape(weights.shape)
            
        if i % report_frequency == 0:             
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)        
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            snapshots.append((np.copy(weights),(pos_correct+neg_correct)/2.0))

    return snapshots
    # return np.array(snapshots)

snapshots = train_graph(pos_examples, neg_examples)

def plotit(pos_examples,neg_examples,snapshots,step):
    fig = pylab.figure(figsize=(10,4))
    fig.add_subplot(1, 2, 1)
    plot_boundary(pos_examples, neg_examples, snapshots[step][0])
    fig.add_subplot(1, 2, 2)
    pylab.plot(np.arange(len(snapshots[:,1])), snapshots[:,1])
    pylab.ylabel('Accuracy')
    pylab.xlabel('Iteration')
    pylab.plot(step, snapshots[step,1], "bo")
    pylab.show()
def pl1(step):
    plotit(pos_examples,neg_examples,snapshots,step)

# 这里并没有出现组件, 感觉是python环境或者版本的问题
interact(pl1, step=widgets.IntSlider(value=0, min=0, max=len(snapshots)-1))
