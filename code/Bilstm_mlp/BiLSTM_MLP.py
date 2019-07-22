# -*- coding: utf-8 -*-
# file: BiLSTM_MLP.py
import sys
import jieba
import pickle
import codecs
import math
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import rnncell as rnn
from collections import Counter
from npdata_provider import data_provider_with_processer
from sklearn import metrics

def segment(sent):
    """
	该函数负责将句子分词
    sent: 需要进行分词的句子
    return: jieba分词后的结果
    """
    # 如果要去停用词，把停用词表做成一个dict，处理速度要快一点
    str_seg = jieba.cut(sent)
    res_str = " ".join(str_seg).split(" ")
    return res_str

def get_stop_words():
    file = codecs.open('data_wjc/baidu_chinese_stopwords.txt', 'r', encoding='utf-8')
    dic_stop_word = {}
    for line in file:
        line = line.strip()
        dic_stop_word[line] = line

    return dic_stop_word

def get_word_dict(sents_list):
    """
    该函数负责构建一个词典
    sents_list: 句子构成的一个list
    return: 分词构成的字典{（word,id）}
    """
    count = []
    word2idx = {}
    max_len = 0
    # print(len(sents_list))
    # 得到停用词表中的所有停用词，存储在remove_word_list中
    dic_stop_word = get_stop_words()
    remove_word_list = []
    for key in dic_stop_word.keys():
        remove_word_list.append(key)
    # 得到数据集中除了停用词之外的所有词
    words = []
    for sent in sents_list:
        tmp = []
        sent = sent.strip()
        str_seg = jieba.cut(sent)
        res_str = " ".join(str_seg).split(" ")
        max_len = max(max_len, len(res_str))
        for word in res_str:
            if word not in remove_word_list:
                tmp.append(word)
        words.extend(tmp)

    # 对全体词进行词频排序
    count.extend(Counter(words).most_common())

    # 按照词频对每个词赋予一个id，词频大的label id值靠前
    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    with open('pkl_data/word_dict.pkl', 'wb') as f:
        pickle.dump(word2idx, f)

    return word2idx, max_len

def get_sents_code(sents_list,word2id,len_sent):
    """
    该函数根据一个词的字典，将句子转换为对应数字的列表
    sents_list: 句子构成的一个list
    word2id：分词构成的字典{（word,id）}
    len_sent：句子的最大长度，如果句子没有该长度就补0
    return: 分词构成的字典{（word,id）}
    """
    sents_code = []
    dic_stop_word = get_stop_words()
    remove_word_list = []
    for key in dic_stop_word.keys():
        remove_word_list.append(key)
    for sent in sents_list:
        tmp = []
        sent = sent.strip()
        str_seg = jieba.cut(sent)
        res_str = " ".join(str_seg).split(" ")
        for word in res_str:
            if word not in remove_word_list:
                tmp.append(word2id[word])
        while True:
            if len(tmp)==len_sent:
                break
            tmp.append(0)
        sents_code.append(tmp)

    return sents_code

def bi_lstm_layer(lstm_inputs, lstm_dim):
    """
    该函数负责将句子用双向LSTM编码，取最后的句子编码
    lstm_inputs: [batch_size, num_steps, emb_size]
    lstm_dim 由于输出结果维度中最后一维是2*lstm_dim，而我们需要的是
    [batch_size, emb_size]
    那么根据这个关系我们可以确定我们需要的lstm_dim会是embedding_size/2
    return: [batch_size, 2*lstm_dim]
    """
    with tf.variable_scope("bi_lstm"):
        lstm_cell = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                    lstm_dim,
                    use_peepholes=True,
                    initializer=tf.random_normal_initializer(stddev=0.1),
                    state_is_tuple=True)
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell["forward"],
            lstm_cell["backward"],
            lstm_inputs,
            dtype=tf.float32)
    # 取出output最后一个时刻句子的embedding结果作为后续输入
    out = tf.concat(outputs, axis=2)
    res = out[:, -1, :]
    return res

X_train, X_test, Y_train, Y_test = data_provider_with_processer(selected_y = 1)
titledata = pd.read_csv('data_wjc/all_title_subtitle.csv', encoding='gbk')
titles = titledata['title'].values
subtitles = titledata['subtitle'].values

all_titles = list(titles) + list(subtitles)
word2id, len_sent = get_word_dict(all_titles)

# titles_code=get_sents_code(titles,word2id)
# subtitles_code=get_sents_code(subtitles,word2id)
# print(titles_code)
# print(subtitles_code)

# word2id_title, len_title_sent = get_word_dict(titles)
# word2id_subtitle, len_subtitle_sent = get_word_dict(subtitles)
# print(len_title_sent, len_subtitle_sent)

# 设置词嵌入的大小
emb_size=int(sys.argv[1])
vocab_size=len(word2id)
col_struct_datas=X_train.shape[1]-2

print("emb size:",emb_size)
print("vocab size:",vocab_size)
print("len of sent:",len_sent)
print("col num of struct datas:",col_struct_datas)

struct_datas=tf.placeholder(tf.float32,[None,col_struct_datas])
y_datas=tf.placeholder(tf.float32,[None])

unstruct_datas_subtitle=tf.placeholder(tf.int32,[None,len_sent])
unstruct_datas_title=tf.placeholder(tf.int32,[None,len_sent])

emb_table=tf.Variable(tf.random_normal([vocab_size,emb_size],stddev=0.05))

title_emb=tf.nn.embedding_lookup(emb_table, unstruct_datas_title)
subtitle_emb=tf.nn.embedding_lookup(emb_table, unstruct_datas_subtitle)
# print(title_emb, subtitle_emb)

# 这里是用的title
bilstm_output=bi_lstm_layer(title_emb,lstm_dim=emb_size)
# print(bilstm_output)
mlp_input=tf.concat(axis=1,values=[struct_datas,bilstm_output])

mlp_input_dim=col_struct_datas+emb_size*2
hidden_dim=mlp_input_dim
print("mlp input dim:",mlp_input_dim)
print("hidden dim:",hidden_dim)

W1=tf.Variable(tf.random_normal([mlp_input_dim,hidden_dim],stddev=0.05))
b1=tf.Variable(tf.random_normal([mlp_input_dim],stddev=0.05))
W2=tf.Variable(tf.random_normal([hidden_dim,hidden_dim],stddev=0.05))
b2=tf.Variable(tf.random_normal([hidden_dim],stddev=0.05))
W3=tf.Variable(tf.random_normal([hidden_dim,1],stddev=0.05))

hidden_units1=tf.sigmoid(tf.matmul(mlp_input,W1)+b1)
hidden_units2=tf.sigmoid(tf.matmul(hidden_units1,W2)+b2)
mlp_output=tf.matmul(hidden_units1,W3)*100
# print(mlp_output)

lr=float(sys.argv[2])
print("lr:",lr)
loss = tf.reduce_mean(tf.square(tf.squeeze(mlp_output)-y_datas))
abs_loss = tf.reduce_mean(tf.abs(tf.squeeze(mlp_output)-y_datas))
# loss=tf.nn.l2_loss(mlp_output-y_datas)
opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

init_var = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_var)

X_train_title=[]
X_train_tmp=X_train[:,-2:-1]
for tmp in X_train_tmp:
    X_train_title.append(tmp[0])

X_train_subtitle=[]
X_train_tmp=X_train[:,-1:]
for tmp in X_train_tmp:
    X_train_subtitle.append(tmp[0])

X_test_title=[]
X_test_tmp=X_test[:,-2:-1]
for tmp in X_test_tmp:
    X_test_title.append(tmp[0])

X_test_subtitle=[]
X_test_tmp=X_test[:,-1:]
for tmp in X_test_tmp:
    X_test_subtitle.append(tmp[0])

# print(X_train_title)
# print(X_train_subtitle)
# print(Y_train)
# print(X_train[:,:-2])

# train
total_train_size = X_train.shape[0]
total_test_size = X_test.shape[0]

batch_size = int(sys.argv[3])
nepoches = 10000
print("batch size:",batch_size)

n_train_batch = math.floor(total_train_size/batch_size)
n_test_batch = math.floor(total_test_size/batch_size)
results=[]
ltime=time.localtime(time.time())
strtime=str(ltime.tm_mon)+str(ltime.tm_mday)+str(ltime.tm_hour)+str(ltime.tm_min)+str(ltime.tm_sec)
for epoch in range(nepoches):
    for i in range(n_train_batch):
        lidx = batch_size*i
        ridx = batch_size*(i+1)
        feed_dict = {
            struct_datas: X_train[lidx:ridx, :-2],
            y_datas: Y_train[lidx:ridx],
            unstruct_datas_title: get_sents_code(X_train_title[lidx:ridx], word2id, len_sent),
            unstruct_datas_subtitle: get_sents_code(X_train_subtitle[lidx:ridx], word2id, len_sent),
        }
        train_loss, train_opt = sess.run([loss, opt],feed_dict=feed_dict)

        if i%5 == 0:
            test_loss=0
            test_abs_loss=0
            for j in range(n_test_batch):
                lidx = batch_size * j
                ridx = batch_size * (j + 1)
                feed_dict = {
                    struct_datas: X_test[lidx:ridx, :-2],
                    y_datas: Y_test[lidx:ridx],
                    unstruct_datas_title: get_sents_code(X_test_title[lidx:ridx], word2id, len_sent),
                    unstruct_datas_subtitle: get_sents_code(X_test_subtitle[lidx:ridx], word2id, len_sent),
                }
                tmp_test_loss, tmp_abs_loss = sess.run([loss, abs_loss], feed_dict=feed_dict)

                test_loss += tmp_test_loss
                test_abs_loss += tmp_abs_loss
            test_loss /= n_test_batch
            test_abs_loss /= n_test_batch

            feed_dict = {
                struct_datas: X_test[:, :-2],
                y_datas: Y_test,
                unstruct_datas_title: get_sents_code(X_test_title, word2id, len_sent),
                unstruct_datas_subtitle: get_sents_code(X_test_subtitle, word2id, len_sent),
            }
            y_pred_tmp = sess.run(mlp_output, feed_dict=feed_dict)

            tmp = []
            tmp2 = []
            for i in range(len(y_pred_tmp)):
                tmp.append(round(y_pred_tmp[i][0]))
                tmp2.append(round(Y_test[i]))
            y_pred = tmp
            y_true = tmp2
            print("--- some pred results:",y_pred[0:10])
            print("--- some true results:",y_true[0:10])

            r2_score = metrics.r2_score(y_pred=y_pred,y_true=y_true)

            print("epoch:",epoch,"batch:",i,"train loss:",round(train_loss,2),"test loss",round(test_loss,2),
                  "abs_loss",round(test_abs_loss,2),"r2 score:",round(r2_score,4))
            result={
                "epoch": epoch, "batch": i, "train loss": train_loss, "test loss": test_loss,
                "abs_loss": test_abs_loss, "r2 score": r2_score
            }
            results.append(result)
            with open('pkl_data/result_%s_%d_%.3f_%d.pkl'%(strtime,emb_size,lr,batch_size), 'wb') as f:
                pickle.dump(results, f)