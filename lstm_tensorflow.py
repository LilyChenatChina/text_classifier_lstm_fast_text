# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 06:49:39 2018

@author: Zheng Yuxing
该文件中定义了LSTM分类器类.
"""
import tensorflow as tf

all_class = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# 定义用于文本分类的LSTM类
class LstmClassifier:

    # 类的初始化函数，该函数用于搭建或者导入计算图
    def __init__(self, class_id, vocabulary_size, embed_mat, is_warm_start):
        self.class_name = all_class[class_id]
        self.vocabulary_size = vocabulary_size
        self.embed_mat = embed_mat

        # LSTM结构中状态向量的维度
        self.lstm_state_fw_size = 128
        self.lstm_state_bw_size = 128
        
        # 网络参数的初始化
        self.stddev = 0.01      # 方差
        self.bias = 0.1         
        
        self.learning_rate = 1e-3  # 学习率

        if is_warm_start:
            # 导入计算图
            tf.reset_default_graph()

            self.sess = tf.Session()
            print('正在加载模型......')
            new_saver = tf.train.import_meta_graph('graph/lstm_model_'+self.class_name+'.ckpt.meta')
            new_saver.restore(self.sess, 'graph/lstm_model_'+self.class_name+'.ckpt')
            print('模型加载完毕！')
            print('--------------------------------------------')

            # 导入操作
            self.output = tf.get_collection('output')[0]
            self.accuracy = tf.get_collection('accuracy')[0]
            self.train_op = tf.get_collection('train_op')[0]

            graph = tf.get_default_graph()
            # 加载占位符
            self.input_x = graph.get_operation_by_name('input_x').outputs[0]
            self.y_label = graph.get_operation_by_name('y_label').outputs[0]
            self.batch_size = graph.get_operation_by_name('batch_size').outputs[0]
            self.state_keep_prob = graph.get_operation_by_name('state_keep_prob').outputs[0]
            self.input_keep_prob = graph.get_operation_by_name('input_keep_prob').outputs[0]
            self.sequence_length = graph.get_operation_by_name('sequence_length').outputs[0]

        else:
            # 定义一个新的计算图
            tf.reset_default_graph()

            print('正在搭建计算图......')
            self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')              # 输入占位符
            self.y_label = tf.placeholder(tf.float32, [None, 1], name='y_label')               # 类标签占位符
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')                  # 批量大小
            self.state_keep_prob = tf.placeholder(tf.float32, name='state_keep_prob')
            self.input_keep_prob = tf.placeholder(tf.float32, name='input_keep_prob')          # Dropout概率
            self.sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')    # 序列长度

            # 定义计算图的各种操作
            self.output = self.lstm()
            self.train_op, self.accuracy = self.graph_operation()

            # 保存各个操作
            tf.add_to_collection('output', self.output)
            tf.add_to_collection('accuracy', self.accuracy)
            tf.add_to_collection('train_op', self.train_op)
            print('计算图搭建完毕！')
            print('--------------------------------------------')

            # 定义会话
            self.sess = tf.Session()
            # 初始化变量
            self.sess.run(tf.global_variables_initializer())

    # 权重与偏置的初始化
    def weight_bias_init(self, shape_w, shape_b):
        init_w = tf.truncated_normal(shape_w, stddev=self.stddev)
        init_b = tf.constant(self.bias, shape=shape_b)
        return tf.Variable(init_w), tf.Variable(init_b)

    # 定义网络
    def lstm(self):
        # 定义全连接层的权重与偏置
        w_h1, b_h1 = self.weight_bias_init([256, 1024], [1024])         # 第一层隐藏层的权重与偏置
        w_h2, b_h2 = self.weight_bias_init([1024, 128], [128])          # 第二层隐藏层的权重与偏置
        w_output, b_output = self.weight_bias_init([128, 1], [1])       # 输出层的权重与偏置

        # 定义前向LSTM核
        lstm_cell_fw = tf.nn.rnn_cell.LSTMCell(self.lstm_state_fw_size, use_peepholes=True)
        lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_fw, input_keep_prob=self.input_keep_prob,
                                                     state_keep_prob=self.state_keep_prob)          # 添加Dropout
        # 定义后向LSTM核
        lstm_cell_bw = tf.nn.rnn_cell.LSTMCell(self.lstm_state_bw_size, use_peepholes=True)
        lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell_bw, input_keep_prob=self.input_keep_prob,
                                                     state_keep_prob=self.state_keep_prob)          # 添加Dropout
    
        # 定义LSTM核的初始化状态
        init_state_fw = lstm_cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        init_state_bw = lstm_cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        # 通过查表法将单词的index转化为FastText词嵌入
        word_vec = tf.nn.embedding_lookup(self.embed_mat, self.input_x, partition_strategy='mod')

        # 双向LSTM网络
        outputs, output_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw, cell_bw=lstm_cell_bw,
                                                                inputs=word_vec, initial_state_fw=init_state_fw,
                                                                initial_state_bw=init_state_bw,
                                                                sequence_length=self.sequence_length)
        # 对双向LSTM网络输出序列进行池化操作
        outputs = tf.concat(values=outputs, axis=2)
        output_max_pool = tf.reduce_max(input_tensor=outputs, axis=1)

        # 全连接层
        h1 = tf.nn.relu(tf.matmul(output_max_pool, w_h1)+b_h1)
        h_drop1 = tf.nn.dropout(h1, keep_prob=self.state_keep_prob)
        h2 = tf.nn.relu(tf.matmul(h_drop1, w_h2)+b_h2)
        h_drop2 = tf.nn.dropout(h2, keep_prob=self.state_keep_prob)
        final_output = tf.sigmoid(tf.matmul(h_drop2, w_output)+b_output)

        return final_output

    # 定义在计算图上的若干操作
    def graph_operation(self):
        y = self.output
        # 交叉熵
        cross_entropy = - tf.reduce_mean(self.y_label * tf.log(tf.clip_by_value(y, 1e-18, 1.0)) +
                                         (1 - self.y_label) * tf.log(tf.clip_by_value((1 - y), 1e-18, 1.0)))
        # 计算分类器精度的操作
        accuracy = tf.reduce_sum(tf.abs(y - self.y_label))
        # 训练计算图的操作
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        return train_op, accuracy

    # 在Session中运行更新操作
    def update(self, batch_text, y_label, batch_size, state_keep_prob, input_keep_prob, sequence_length):
        self.sess.run(self.train_op, feed_dict={self.input_x: batch_text,
                                                self.y_label: y_label,
                                                self.batch_size: batch_size,
                                                self.state_keep_prob: state_keep_prob,
                                                self.input_keep_prob: input_keep_prob,
                                                self.sequence_length: sequence_length})

    # 在Session中运行计算精度的操作
    def cal_accuracy(self, batch_text, y_label, batch_size, sequence_length):
        accuracy = self.sess.run(self.accuracy, feed_dict={self.input_x: batch_text,
                                                           self.y_label: y_label,
                                                           self.batch_size: batch_size,
                                                           self.state_keep_prob: 1.0,
                                                           self.input_keep_prob: 1.0,
                                                           self.sequence_length: sequence_length})
        return accuracy

    # 在Session中运行预测操作
    def predict(self, batch_text, batch_size, sequence_length):
        result = self.sess.run(self.output, feed_dict={self.input_x: batch_text,
                                                       self.batch_size: batch_size,
                                                       self.state_keep_prob: 1.0,
                                                       self.input_keep_prob: 1.0,
                                                       self.sequence_length: sequence_length})
        return result

    # 保存计算图
    def save_graph(self):
        print('正在保存模型.........')
        saver = tf.train.Saver()
        saver.save(self.sess, 'graph/lstm_model_'+self.class_name+'.ckpt')
        print('模型保存完毕！')
        print('------------------------------------------------')
