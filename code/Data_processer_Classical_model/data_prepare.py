import re
import sqlite3
import json, pickle
import unicodecsv as csv
import pandas as pd
import numpy as np


df_middle = pd.read_csv("data/csv/integral_data.csv")
df_excp = pd.read_csv("data/csv/except_data.csv")
df_station = pd.read_csv("data/metros_geo_ok.csv")
np_sta_latb = df_station['LATB'].values
np_sta_lngb = df_station['LNGB'].values

with open('data/dis_scale', 'rb') as f:
	col_content = pickle.load(f)


# final selected features
select_features = [
	# 'url', 
	'unit_price', 
	'total_price', 
	# 'community', 'community_url',
	'district', # one-hot ['嘉定','松江','宝山', '浦东','普陀','徐汇','闵行','静安','虹口','杨浦','黄浦','金山','奉贤','长宁','青浦','崇明','闸北']
	# 'location', 
	# 'huxing', 
	# 'floor', 
	# 'building_area',
	# 'available_area', 
	# 'huxing_type', 		# Due to low distinguishablity ['平层','跃层','复式','错层']
	# 'face_to', 
	'building_structure', 	# one-hot [ '板楼', '塔楼', '板塔结合', '平房']
	'building_type', 		# one-hot ['钢混结构', '砖混结构',  '未知结构',  '混合结构', '框架结构', '砖木结构', '钢结构']
	'decoration', 			# low distinguishablity, one-hot ['精装', '简装', '毛坯', '其他']
	# 'elevator_rate', 	
	# 'elevator',
	# 'property_year',		# sparse feature 
	'trade_type', 			# very low distinguishablity, can choose to be closed, or one-hot ['商品房', '售后公房', '动迁安置房']
	'house_usage', 			# one-hot encode ['普通住宅', '别墅', '商业办公类', '老公寓', '花园洋房', '新式里弄', '旧式里弄']
	# 'limit_year',			# sparse feature
	# 'property_belong',	# extremely low distinguishablity ['共有','非共有']
	# 'mortgage', 			# not ready feature
	'tags', 'feature', 'title', 'subtitle',
	'dist_sta', 			# float
	'dist_cent', 			# float
	'latitude', 'longitude', 
	'huxing_shi',			# int
	'huxing_ting', 			# int
	'huxing_chu', 			# int
	'huxing_wei', 			# int
	'floor_float',			# layer number, float
	'building_area_float', 	# float
	'face_N', 'face_S', 'face_E', 'face_W',	# binary
	'elevator_01', 'elevator_rate_float'	# binary and float
]

# new feature name for final csv file of training
varibale_feature = [
	'unit_price', 
	'total_price',
	'building_area_float', 	# float
	'dist_sta', 			# float
	'dist_cent', 			# float 
	'huxing_shi',			# int
	'huxing_ting', 			# int
	'huxing_chu', 			# int
	'huxing_wei', 			# int
	'floor_float',			# layer number, float
	'face_N', 'face_S', 'face_E', 'face_W',	# binary
	'elevator_01', 'elevator_rate_float'	# binary and float
]

# one-hot encoding
df_select = df_middle[select_features].copy()
df_district = pd.get_dummies(df_select[['district']], prefix = 'district')
df_building_str = pd.get_dummies(df_select['building_structure'], prefix = 'building_structure')
df_building_type = pd.get_dummies(df_select['building_type'], prefix = 'building_type')
df_trade_type = pd.get_dummies(df_select['trade_type'], prefix = 'trade_type')
df_house_usage = pd.get_dummies(df_select['house_usage'], prefix = 'house_usage')

# for classical model
df_all = pd.concat([ df_select[varibale_feature], df_district, df_building_str, df_building_type, df_trade_type, df_house_usage ], axis = 1)

# for BiLSTM model
df_all_title_subtitle = pd.concat([df_all, df_middle[['title', 'subtitle']]], axis = 1)

df_all.to_csv("data/csv/final_training_data.csv",index = False)
df_all_title_subtitle.to_csv("data/csv/all_title_subtitle.csv", index = False)

# map data used to draw map distribution plot
df_map_data = pd.concat([df_select[['unit_price', 'latitude', 'longitude']]], axis =1)
