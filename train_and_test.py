# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 06:49:39 2018

@author: Zheng Yuxing
这是运行网络评论分类器的主程序，
train()函数用于训练该LSTM分类器，
test()函数用于使用已经训练好的分类器对测试集数据进行分类

由于这里存在6个相互独立的类别，因此这里训练了六个LSTM对这六个类别分别分类
"""

from lstm_tensorflow import LstmClassifier
import text_process
import numpy as np
import time

# 类别名
class_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

num_words = 160000
b_size = 100


# 训练函数
def train(embed_mat, is_warm_start, class_id, num_step=1000):
    # 样本预处理
    positive_sample, negative_sample, p_length, n_length = text_process.create_sample(class_id=class_id,
                                                                                      num_words=num_words)
    # 生成分类器实体
    text_classifier = LstmClassifier(class_id=class_id, vocabulary_size=num_words+1,
                                     embed_mat=embed_mat, is_warm_start=is_warm_start)

    print('正在训练', class_name[class_id], '类别的分类器')
    print('开始训练.......')
    print('----------------------------------------------------')
    start = time.clock()
    for step in range(num_step):
        # 生成小批量样本
        batch_data = text_process.create_batch(positive_sample=positive_sample, negative_sample=negative_sample,
                                               n_length=n_length, p_length=p_length, b_size=b_size)
        batch_sample = batch_data[0]
        batch_label = batch_data[1]
        sequence_length = batch_data[2]
        # 迭代更新
        text_classifier.update(batch_text=batch_sample, y_label=batch_label, batch_size=2*b_size,
                               state_keep_prob=0.3, input_keep_prob=0.4, sequence_length=sequence_length)
        if((step+1) % 10) == 0:
            # 计算精度
            accuracy = text_classifier.cal_accuracy(batch_text=batch_sample, batch_size=2*b_size,
                                                    sequence_length=sequence_length,
                                                    y_label=batch_label)
            end = time.clock()
            print('已经训练了', int(step+1), '次')
            print('本十次训练用时', end-start, '秒钟')
            print('训练集上误差：', accuracy)
            start = time.clock()

    print('训练结束！')
    print('----------------------------------------------------')

    print('正在保存模型.....')
    text_classifier.save_graph()
    print('模型保存完毕！')
    print('----------------------------------------------------')


# 测试函数
def test(embed_mat):
    test_set, sequence_length, text_id, batch_length = text_process.load_test_set(num_words=num_words)
    result = []

    print('正在预测......')
    for class_id in range(6):
        text_classifier = LstmClassifier(class_id=class_id, embed_mat=embed_mat,
                                         vocabulary_size=num_words+1, is_warm_start=True)
        start = time.clock()
        # 将测试集数据分批导入
        temp_result = []
        for data, length, batch_size in zip(test_set, sequence_length, batch_length):
            output = text_classifier.predict(batch_text=data, batch_size=batch_size, sequence_length=length)
            output = list(np.concatenate(output))
            temp_result = temp_result + output
        result.append(temp_result)
        end = time.clock()
        print(class_name[class_id], '分类器预测完毕！')
        print('预测用时：', end-start, '秒钟')
        print('----------------------------------------------------')
    print('预测完毕！')
    print('----------------------------------------------------')

    # 输出文件
    print('正在输出文件.......')
    text_process.output_csv(result, text_id)
    print('文件输出完毕！')
    print('----------------------------------------------------')


# 运行整个训练与分类的过程
if __name__ == '__main__':
    text_process.create_dict(num_words=num_words)                  # 创建字典
    embedding_mat = text_process.create_word_vec(num_words)        # 生成词向量矩阵
    for class_index in range(6):
        train(embedding_mat, is_warm_start=False, class_id=class_index, num_step=1500)
    test(embedding_mat)

