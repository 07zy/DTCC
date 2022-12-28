# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf
import numpy as np
import sys, time
from scipy.special import comb
from sklearn import metrics
import copy
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from scipy.special import comb
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans



def ri_score(y_true, y_pred):
    tp_plus_fp = comb(np.bincount(y_true), 2).sum()
    tp_plus_fn = comb(np.bincount(y_pred), 2).sum()
    A = np.c_[(y_true, y_pred)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(y_true))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

def nmi_score(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)# average_method='arithmetic'

def cluster_using_kmeans(embeddings, K):
    return KMeans(n_clusters=K).fit_predict(embeddings)



def transfer_labels(labels):
    indexes = np.unique(labels)#该函数是去除数组中的重复数字，并进行排序之后输出。
    num_classes = indexes.shape[0]#2
    num_samples = labels.shape[0]#28
    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indexes)[0][0]#np.argwhere:返回非0的数组元组的索引，其中a是要索引数组的条件。
        labels[i] = new_label
    return labels, num_classes



def _rnn_reformat(x, input_dims, n_steps):
    """
    This function reformat input to the shape that standard RNN can take. 
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])#perm:控制转置的操作,以perm = [0,1,2] ,如果换成[1,0,2],就是把最外层的两维进行转置，比如原来是2乘3乘4，经过[1,0,2]的转置维度将会变成3乘2乘4
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])#把x_的维度变成(n_steps*batch_size, input_dims)

    x_reformat = tf.split(x_, n_steps, 0) # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    return x_reformat


def _rnn_reformat_denoise(x, input_dims, n_steps, batch_size):
    """
    This function reformat input to the shape that standard RNN can take. 
    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    # x_ = x +  np.random.uniform(-1,1,(batch_size, n_steps, input_dims))
    # x_ = x +  np.random.normal(size=(batch_size, n_steps, input_dims))
    # permute batch_size and n_steps
    x_ = tf.transpose(x, [1, 0, 2])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = tf.reshape(x_, [-1, input_dims])
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = tf.split(x_, n_steps, 0)
    return x_reformat


def load_data(filename):
    data_label = np.loadtxt(filename)
    #data_label = pd.read_csv(filename, sep='\t')
    data_label=np.array(data_label)
    data = data_label[:, 1:]
    label =data_label[:, 0].astype(np.int32)#type(label) <class 'numpy.ndarray'>
    return data, label


def evaluation(prediction, label):
    acc = cluster_acc(label, prediction)#prediction=km_idx, label=testing_label
    nmi = metrics.normalized_mutual_info_score(label, prediction)
    ari = metrics.adjusted_rand_score(label, prediction)
    ri = rand_index_score(label, prediction)

    return ri, nmi,ari, acc


def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()#取从n个项目中选择k个项目(不重复且无顺序)的方法数量。
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)#ndarray(28,)y_true=label,y_pred=prediction
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)#shape(2,2)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def next_batch(batch_size, data):
    # assert data.shape[0] == label.shape[0]
    row = data.shape[0]#data(28,286)
    batch_len = int(row / batch_size)
    left_row = row - batch_len * batch_size

    for i in range(batch_len):
        batch_input = data[i * batch_size: (i + 1) * batch_size, :]
        yield (batch_input, False)#yield就是 return 返回一个值，并且记住这个返回的位置，下次迭代就从这个位置后开始。

    if left_row != 0:
        need_more = batch_size - left_row
        need_more = np.random.choice(np.arange(row), size=need_more)
        yield (np.concatenate((data[-left_row:, :], data[need_more]), axis=0), True)


