#!/usr/bin/env python
# -*- coding: utf8 -*-

# author: xiaofengfeng
# create: 2017-03-16 11:12:23

#############################################################################
# C4.5
# Powered by Python 3.
# Code by 小村长: http://blog.csdn.net/Lu597203933/article/details/38024239
# Principle by x454045816: http://m.blog.csdn.net/article/details?id=44726921
#############################################################################

"""
训练集:

    outlook    temperature    humidity    windy
    ---------------------------------------------------------
    sunny       hot            high         false         N
    sunny       hot            high         true          N
    overcast    hot            high         false         Y
    rain        mild           high         false         Y
    rain        cool           normal       false         Y
    rain        cool           normal       true          N
    overcast    cool           normal       true          Y

测试集
    outlook    temperature    humidity    windy
    ---------------------------------------------------------
    sunny       mild           high         false
    sunny       cool           normal       false
    rain        mild           normal       false
    sunny       mild           normal       true
    overcast    mild           high         true
    overcast    hot            normal       false
    rain        mild           high         true
"""

import math


def get_data_set():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    data_set = [[0, 0, 0, 0, 'N'],
                [0, 0, 0, 1, 'N'],
                [1, 0, 0, 0, 'Y'],
                [2, 1, 0, 0, 'Y'],
                [2, 2, 1, 0, 'Y'],
                [2, 2, 1, 1, 'N'],
                [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return data_set, labels


def get_test_set():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    test_set = [[0, 1, 0, 0],
                [0, 2, 1, 0],
                [2, 1, 1, 0],
                [0, 1, 1, 1],
                [1, 1, 0, 1],
                [1, 0, 1, 0],
                [2, 1, 0, 1]]
    return test_set


def calc_shannon_ent(data_set):
    """
    计算数据集香农熵
    """
    num_entries = len(data_set)
    label_counts = dict()
    for feat_vec in data_set:
        current_label = feat_vec[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1

    shannon_ent = float()

    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * math.log(prob, 2)

    return shannon_ent


def split_data_set(data_set, axis, value):
    """
    依据特征划分数据集 axis代表第几个特征 value代表该特征所对应的值 返回的是划分后的数据集
    """
    ret_data_set = list()
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_data_set.append(reduced_feat_vec)
    return ret_data_set


def choose_best_feature_to_split(data_set):
    """
    选择最好的数据集(特征)划分方式  返回最佳特征下标
    """
    num_entries = len(data_set)
    num_features = len(data_set[0]) - 1  # 特征个数
    base_entropy = calc_shannon_ent(data_set)
    best_info_gain = float()
    best_feature = -1
    for i in range(num_features):  # 遍历特征
        feature_set = set([example[i] for example in data_set])  # 第i个特征取值集合
        new_entropy = float()
        for value in feature_set:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / num_entries
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 该特征划分所对应的entropy(信息熵)

        # 计算信息增益值 选择增益值最大的一个
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature


def majority_cnt(class_list):
    """
    多数表决的方法决定叶子节点的分类 当所有的特征全部用完时仍属于多类
    """
    class_count = dict()
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    # sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)  # 排序函数 operator中的
    sorted_class_count = sorted(class_count.items(), key=lambda item: item[1], reverse=True)  # 排序函数 operator中的
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    """
    创建树 python中用字典类型来存储树的结构 返回字典
    """
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):  # 类别完全相同则停止继续划分  返回类标签-叶子节点
        return class_list[0]
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)  # 遍历完所有的特征时返回出现次数最多的

    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label: dict()}
    del(labels[best_feat])

    feat_values = [example[best_feat] for example in data_set]  # 得到的列表包含所有的属性值
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set(data_set, best_feat, value), sub_labels)
    return my_tree


def classify(input_tree, feat_labels, test_vec):
    """
    使用决策树执行分类
    """
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)  # index方法查找当前列表中第一个匹配first_str变量的元素的索引
    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def classifyAll(inputTree, featLabels, testDataSet):
    """
    输入：决策树，分类标签，测试数据集
    输出：决策结果
    描述：跑决策树
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


if __name__ == '__main__':
    data_set, labels = get_data_set()
    labels_tmp = labels[:]
    # shannon_ent = calc_shannon_ent(data_set)
    desicion_tree = create_tree(data_set, labels)
    print("desicion tree:", desicion_tree)

    test_set = get_test_set()
    classify_result = classifyAll(desicion_tree, labels_tmp, test_set)

    print("classify result:")
    for data, result in zip(test_set, classify_result):
        print(data, result)
