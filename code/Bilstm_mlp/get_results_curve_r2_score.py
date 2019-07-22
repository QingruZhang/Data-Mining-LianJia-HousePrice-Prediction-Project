# file: get_results_curve_r2_score.py

import numpy as np
import matplotlib.pyplot as plt
import pickle

# 该脚本负责获取BiLSTM+MLP的收敛过程曲线图

pkl_file150 = open('pkl_data/result_62792613_150_0.008_8000.pkl', 'rb')
pkl_file100 = open('pkl_data/result_62792917_100_0.010_8000.pkl', 'rb')
pkl_file40 = open('pkl_data/result_62793343_40_0.005_8000.pkl', 'rb')
pkl_file20 = open('pkl_data/result_627161833_20_0.001_8000.pkl', 'rb')
pkl_file10 = open('pkl_data/result_62793112_10_0.010_8000.pkl', 'rb')

results150 = pickle.load(pkl_file150)
results100 = pickle.load(pkl_file100)
results40 = pickle.load(pkl_file40)
results20 = pickle.load(pkl_file20)
results10 = pickle.load(pkl_file10)

maxstep=100

x=[1+i for i in range(maxstep)]
y150=[]
y100=[]
y40=[]
y20=[]
y10=[]

for result in results150[0:maxstep]:
    y150.append(result['r2 score'])

for result in results100[0:maxstep]:
    y100.append(result['r2 score'])

for result in results40[0:maxstep]:
    y40.append(result['r2 score'])

for result in results20[0:maxstep]:
    y20.append(result['r2 score'])

for result in results10[0:maxstep]:
    y10.append(result['r2 score'])


plt.plot(x, y150, color='green', label='emb_size=150')
plt.plot(x, y100, color='red', label='emb_size=100')
plt.plot(x, y40, color='skyblue', label='emb_size=40')
plt.plot(x, y20, color='blue', label='emb_size=20')
plt.plot(x, y10, color='yellow', label='emb_size=10')
plt.title('R2 Score of BiLSTM and MLP')
plt.legend()  # 显示图例
plt.xlabel('step')
plt.ylabel('R2 score')
plt.show()