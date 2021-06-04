# 作者：Xiang Zhaoyi
# 日期：2021/6/4 下午4:01
# 工具：PyCharm
# 将所有的csv的文件转换成pkl文件,用于训练
import pickle
import numpy as np
import os
import pandas as pd
from XZY.config import pkl_csv_map, pkl_node


def print_skeleton_info(info: pd.DataFrame):
    """
    打印骨架数据的信息,所有类别的数目
    info: DataFrame
    """
    activity = dict()
    for index, row in info.iterrows():
        if row['activity'] not in activity.keys():
            activity[row['activity']] = 0
        activity[row['activity']] += 1
    for key, value in activity.items():
        print(f'num of {key} : {value}')


def get_csv_data_per_activity(csv_dir):
    # region    #####################################################
    # 将所有的csv读取到all_csv_info中
    all_csv_info = None
    for csv_name in os.listdir(csv_dir):
        if not csv_name.endswith('.csv'):
            continue
        csv_path = os.path.join(csv_dir, csv_name)
        if 'static' in csv_path and 'delete.csv' not in csv_name:  # 针对师兄做的数据集而采取的结果
            continue
        csv_info = pd.read_csv(csv_path, index_col=0)[0:]
        if all_csv_info is None:
            all_csv_info = csv_info
        else:
            all_csv_info = all_csv_info.append(csv_info)
    all_csv_info = all_csv_info.reset_index(drop=True)  # 将index转成序号0.1.2...
    # print(f"num of {all_csv_info['activity']} is {all_csv_info.shape[0]}")
    return all_csv_info
    # endregion #####################################################


if __name__ == '__main__':
    csv_dir_list = [
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/摔倒",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/正常站立",

        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/back_against/back_against_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/climb/climb_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/fall_behind/fall_behind_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/fall_toward/fall_toward_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/hand_out/hand_out_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/head_and_hand_out/head_and_hand_out_static",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/normal_standing",

    ]
    all_csv_info = None
    for csv_dir in csv_dir_list:
        csv_info_per_activity = get_csv_data_per_activity(csv_dir)  # 获得每一个路径下的所有csv信息
        if all_csv_info is None:
            all_csv_info = csv_info_per_activity
        else:
            all_csv_info = all_csv_info.append(csv_info_per_activity)

    # region    #####################################################
    # 将all_csv_info转换成pkl
    pkl_list_csv = []
    for i in pkl_node:
        pkl_list_csv.append(pkl_csv_map[i[0:-1]] + i[-1])
    pkl_list_csv.append('activity')
    pkl_info = all_csv_info[pkl_list_csv]
    pkl_info.columns = pkl_node + ['activity']

    pkl_info.to_pickle('all_skeleton_data.pkl')
    # endregion #####################################################
    pkl_info1 = pickle.load(open('all_skeleton_data.pkl', 'rb'))
    print_skeleton_info(pkl_info)
    print(f"num of all data is : {pkl_info.shape[0]}\tsave pkl successfully !!! ")
