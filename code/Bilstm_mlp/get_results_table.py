# file: get_results_table.py

import pickle
import numpy as np
# file: get_results_table.py

# 该脚本负责获取BiLSTM+MLP的收敛结果表

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

max_r2=-2
min_abs=1000
for result in results150:
    if result['r2 score']>max_r2:
        max_r2=result['r2 score']
        min_abs=result['abs_loss']
print("emb size 150")
print("max r2 score:",max_r2)
print("min abs:",min_abs)
print("")

max_r2=-2
min_abs=1000
for result in results100:
    if result['r2 score']>max_r2:
        max_r2=result['r2 score']
        min_abs=result['abs_loss']
print("emb size 100")
print("max r2 score:",max_r2)
print("min abs:",min_abs)
print("")

max_r2=-2
min_abs=1000
for result in results40:
    if result['r2 score']>max_r2:
        max_r2=result['r2 score']
        min_abs=result['abs_loss']
print("emb size 40")
print("max r2 score:",max_r2)
print("min abs:",min_abs)
print("")

max_r2=-2
min_abs=1000
for result in results20:
    if result['r2 score']>max_r2:
        max_r2=result['r2 score']
        min_abs=result['abs_loss']
print("emb size 20")
print("max r2 score:",max_r2)
print("min abs:",min_abs)
print("")

max_r2=-2
min_abs=1000
for result in results10:
    if result['r2 score']>max_r2:
        max_r2=result['r2 score']
        min_abs=result['abs_loss']
print("emb size 10")
print("max r2 score:",max_r2)
print("min abs:",min_abs)
print("")

