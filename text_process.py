# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 06:49:39 2018

@author: Zheng Yuxing
该文件包括原始文本的预处理的各个函数
为了解决训练集数据样本不平衡的问题，生成小批量数据函数create_batch()中对正例与负例单独采样
这里使用的预训练的FastText词向量数据来源于Kaggle
"""
from keras.preprocessing import text
from keras.preprocessing import sequence
import pandas as pd
import numpy as np
import random as rd


# 类名列表
class_name = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 最大序列长度
max_len = 300


# 创建字典
def create_dict(num_words):
    print('----------------------------------------------------')
    print('正在创建字典.......')
    
    # 读入训练集数据
    text_df = pd.read_csv('data/train.csv')
    train_text = text_df['comment_text']

    # 使用Tokenizer创建字典
    tokenizer = text.Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(train_text)

    # 保存已创建的字典
    f = open('data/word_dict.txt', 'w', encoding='utf-8')
    f.write(str(tokenizer.word_index))
    f.close()
    
    print('字典创建完毕！')
    print('----------------------------------------------------')
    print('\n')


# 导入训练集数据并对其进行预处理
def create_sample(class_id, num_words):
    print('----------------------------------------------------')
    print('正在对原始文本进行预处理.......')
    text_df = pd.read_csv('data/train.csv')

    tokenizer = text.Tokenizer(num_words=num_words)

    # 导入字典
    f = open('data/word_dict.txt', 'r', encoding='utf-8')
    word_dict = f.read()
    word_dict = eval(word_dict)
    tokenizer.word_index = word_dict

    # 划分正类与负类
    positive_text = text_df['comment_text'][text_df[class_name[class_id]] == 1]
    negative_text = text_df['comment_text'][text_df[class_name[class_id]] == 0]

    # 将文本转换为词id的序列
    positive_sample = tokenizer.texts_to_sequences(positive_text)
    negative_sample = tokenizer.texts_to_sequences(negative_text)

    # 记录各个序列长度，最大长度强行设置为300
    p_length = [min(len(p_item), max_len) for p_item in positive_sample]
    n_length = [min(len(n_item), max_len) for n_item in negative_sample]

    print('文本处理完毕！')
    print('----------------------------------------------------')

    return positive_sample, negative_sample, p_length, n_length


# 导入测试集数据并对其进行预处理
def load_test_set(num_words):
    print('----------------------------------------------------')
    print('正在对测试集文本进行预处理.......')
    test_text_df = pd.read_csv('data/test.csv')

    tokenizer = text.Tokenizer(num_words=num_words)

    # 导入字典
    f = open('data/word_dict.txt', 'r', encoding='utf-8')
    word_dict = f.read()
    word_dict = eval(word_dict)
    tokenizer.word_index = word_dict

    # 将原始文本转换成词id序列
    test_text = tokenizer.texts_to_sequences(test_text_df['comment_text'])
    text_id = test_text_df['id']
    batch_num = int(len(text_id)/200)

    # 将测试集数据分批
    test_set = []
    batch_length = []
    sequence_length = []
    for i in range(batch_num):
        length = [min(len(item), max_len) for item in test_text[i*200:(i+1)*200]]
        sequence_length.append(length)
        test_set.append(sequence.pad_sequences(test_text[i*200:(i+1)*200], maxlen=max(length),
                                               padding='post', truncating='post'))
        batch_length.append(len(test_text[i*200:(i+1)*200]))
    length = [min(len(item), max_len) for item in test_text[batch_num*200:]]
    sequence_length.append(length)
    test_set.append(sequence.pad_sequences(test_text[batch_num*200:], maxlen=max(length),
                                           padding='post', truncating='post'))
    batch_length.append(len(test_text[batch_num*200:]))

    print('文本处理完毕！')
    print('----------------------------------------------------')

    return test_set, sequence_length, text_id, batch_length


# 生成小批量(batch)样本
def create_batch(positive_sample, negative_sample, p_length, n_length, b_size):
    # 随机抽取正类样本
    random_id = rd.sample(range(len(positive_sample)), b_size)
    batch_half = [positive_sample[i] for i in random_id]
    sequence_length_half = [p_length[i] for i in random_id]
    # 随机抽取负类样本
    random_id = rd.sample(range(len(negative_sample)), b_size)
    batch = batch_half + [negative_sample[i] for i in random_id]
    sequence_length = sequence_length_half + [n_length[i] for i in random_id]

    # 将生成的小批量数据处理成同样的长度
    length = max(sequence_length)
    batch = sequence.pad_sequences(batch, maxlen=length, padding='post', truncating='post')
    # 贴标签
    batch_label = np.concatenate((np.ones((b_size, 1)), np.zeros((b_size, 1))))

    return batch, batch_label, sequence_length


# 输出预测结果
def output_csv(result, text_id):
    print('正在输出文件.......')

    data_item = pd.DataFrame()
    data_item['id'] = text_id
    for name, item in zip(class_name, result):
        data_item[name] = item
    data_item.to_csv('data/lstm_result.csv', index=False)

    print('输出文件完成！')
    print('----------------------------------------------------')


# 生成由FastText词向量组成的矩阵
def create_word_vec(num_words=10000):
    embed_mat = np.zeros((num_words+1, 300), dtype='float32')

    # 导入字典
    f = open('data/word_dict.txt', 'r', encoding='utf-8')
    word_dict = f.read()
    word_dict = eval(word_dict)
    sub_dict = {k: v for k, v in word_dict.items() if v <= num_words}   # 舍弃字典中的低频词
    
    # 打开FastText词向量文件
    vecs = open('data/crawl-300d-2M.vec')
    count = 0
    # 将FastText词向量与字典中的词汇对应
    for item in vecs:
        word = item.rstrip().rsplit(' ')[0]
        if word in sub_dict.keys():
            embed_mat[sub_dict[word]] = np.asarray(item.rstrip().rsplit(' ')[1:], dtype=np.float32)
            count = count+1
        if count >= num_words:
            break
            
    return embed_mat


# 测试该模块时使用
if __name__ == '__main__':
    create_dict(20000)
    # p_sample, n_sample, pl, nl = create_sample(0, 10000)
    # b, b_label, s_length = create_batch(p_sample, n_sample, pl, nl, 100)
    # print(b)
    # print(b_label)
    # print(s_length)
    create_word_vec(20000)
