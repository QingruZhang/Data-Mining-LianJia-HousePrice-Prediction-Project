import re
import sqlite3
import json, pickle
import unicodecsv as csv
import pandas as pd
import numpy as np
import util 
from util import chinese_to_arabic

CNETER_LATITUDE = 31.2304
CNETER_LONGTITUDE = 121.4737
RADIOUS_EARTH = 1

con = sqlite3.connect("data/lianjia/lianjia.db")
cur = con.execute("select * from house_detail")
col_name_list = [tmp[0] for tmp in cur.description]

# process floor text
def get_floor_float(floor):
    if floor[7].isdigit():
        overall = int(floor[6:8])
    else:
        overall = int(floor[6])
    if floor[0] == '低':
        return 0
    elif floor[0] == '中':
        return overall * 0.5    
    else:
        return overall

# unit funciton to process Chinese number
def get_int_from_CH(chstr):
    CH_letter = ['一','二','三','四','五','六','七','八','九','十']
    res = CH_letter.index(chstr)+1 if chstr != '百' else 100
    return res

# process elevator rate text
def get_elevator_rate(chstr):
    # use function form util.py
    num = chinese_to_arabic(chstr[0])
    all_int = chinese_to_arabic(chstr[2:-1])
    return num / all_int

# count value scale for every attribute 
col_content = {}
for key in col_name_list:
    col_content[key] = []

# discrete attribute
dis_val_list = [
    'community',
    'district',
    'location',
    'huxing',
    'floor',
    'huxing_type',
    'face_to',
    'building_structure',
    'building_type',
    'decoration',
    'elevator_rate',
    'elevator',
    'elevator',
    # 'roperty_year',
    'trade_type',
    'house_usage',
    # 'limit_year',
    'property_belong',
    'mortgage',
]


refresh = False
refresh2 = True

# count value scale for every attribute and save it 
if refresh:
    j = 0
    for line in cur:
        if j % 10000 == 0:
            print(j)
        j += 1
        for i in range(28):
            if col_name_list[i] in dis_val_list:
                feature = col_name_list[i]
                if line[i] not in col_content[feature]:
                    col_content[feature].append(line[i])
    with open("data/dis_scale", 'wb') as f:
        pickle.dump(col_content, f)
else:
    with open('data/dis_scale', 'rb') as f:
        col_content = pickle.load(f)

# station dataFram
df_station = pd.read_csv("data/metros_geo_ok.csv")
np_sta_latb = df_station['LATB'].values
np_sta_lngb = df_station['LNGB'].values

# selected attributes
target_cloumns = col_name_list + [
    'dist_sta','dist_cent', 'latitude', 'longitude', 
    'huxing_shi', 'huxing_ting','huxing_chu','huxing_wei',
    'floor_float', 'building_area_float', 
    'face_N','face_S','face_E','face_W',
    'elevator_01', 'elevator_rate_float',
]

# Data process
if refresh2:
    with open("data/csv/integral_data.csv", 'wb') as f:
        # wrong data file
        f_excp = open('data/csv/except_data.csv', 'wb')
        excp_csv_file = csv.writer(f_excp)
        excp_csv_file.writerow(target_cloumns)

        # normal data file
        csv_file = csv.writer(f)
        cur = con.execute("select * from house_detail")
        csv_file.writerow(target_cloumns)
        num = 0
        for line in cur:
            line = list(line)
            tmp_dict = dict(zip(col_name_list, line))
            if num % 10000 == 0:
                print(num)
            num += 1
            # judge attributed whether correct
            if tmp_dict['building_structure'] in ['暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['building_type'] in ['暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['decoration'] not in ['精装','简装','毛坯','其他']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['district'] is None:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['elevator'] not in ['有', '无']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['elevator_rate'] in [None, '暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['face_to'] in ['毛坯','其他','精装','简装','暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['house_usage'] in ['暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['huxing_type'] not in ['平层','跃层','复式','错层',]:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['location'] is None:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['mortgage'] in ['暂无数据']:
                excp_csv_file.writerow(line)
                continue
            elif tmp_dict['property_belong'] in ['暂无数据']:
                excp_csv_file.writerow(line)
                continue
            else:
                # process attribute to target form
                url_comm = line[4]
                latitude, longitude = con.execute("select longitude,latitude from community where url == '%s'"%url_comm).fetchall()[0]
                dist_sat = np.min(np.abs(np_sta_latb - latitude) + np.abs(np_sta_lngb - longitude) ) * RADIOUS_EARTH
                dist_cent = (abs(latitude - CNETER_LATITUDE) + abs(longitude - CNETER_LONGTITUDE) ) * RADIOUS_EARTH

                hux_list = re.split('室|厅|厨|卫' , tmp_dict['huxing'])

                huxing_1 = int(hux_list[0])
                huxing_2 = int(hux_list[1])
                huxing_3 = int(hux_list[2])
                huxing_4 = int(hux_list[3])

                floor_float = get_floor_float(tmp_dict['floor'])
                building_area_float = float(tmp_dict['building_area'][:-1])
                face_S = 1 if '南' in tmp_dict['face_to'] else 0
                face_N = 1 if '北' in tmp_dict['face_to'] else 0
                face_E = 1 if '东' in tmp_dict['face_to'] else 0
                face_W = 1 if '西' in tmp_dict['face_to'] else 0
                elevator_01 = 1 if tmp_dict['elevator'] == '有' else 0
                elevator_rate = get_elevator_rate(tmp_dict['elevator_rate'])

                # wirte to normal file
                csv_file.writerow(line + [dist_sat, dist_cent, latitude, longitude, huxing_1, huxing_2, \
                            huxing_3, huxing_4, floor_float, building_area_float, face_N, face_S, \
                            face_E, face_W, elevator_01, elevator_rate ])
            
        f_excp.close()

df_middle = pd.read_csv("data/csv/integral_data.csv")
df_excp = pd.read_csv("data/csv/except_data.csv")

con.close()



