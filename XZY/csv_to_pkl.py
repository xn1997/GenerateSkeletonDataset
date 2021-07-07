# 作者：Xiang Zhaoyi
# 日期：2021/6/4 下午4:01
# 工具：PyCharm
# 将所有的csv的文件转换成pkl文件,用于训练
import pickle
import numpy as np
import os
import pandas as pd
from XZY.libs.config import pkl_csv_map, pkl_node


def print_skeleton_info(info: pd.DataFrame, show_flag=True):
    """
    打印骨架数据的信息,所有类别的数目
    info: DataFrame
    show_flag: True显示多少类,False:统计多少类
    """
    activity = dict()
    for index, row in info.iterrows():
        if row['activity'] not in activity.keys():
            activity[row['activity']] = 0
        activity[row['activity']] += 1
    if show_flag:
        for key, value in activity.items():
            print(f'num of {key} : {value}')
    return activity


def get_train_val(info: pd.DataFrame, train_ratio=1.0, random_seed=None):
    """
    从所有的数据中划分训练集和验证集
    random_seed: 随机种子。None表示不使用随机抽样
    """
    # region 分别读取每一类的数据
    all_class_info = dict()
    cls = print_skeleton_info(info, show_flag=False)
    for key, value in cls.items():
        index = info['activity'] == key
        all_class_info[key] = info[index]  # .head()
    # endregion

    # region 划分数据集
    train_data = None
    val_data = None
    for key, value in all_class_info.items():
        value = value.reset_index(drop=True)
        end_index = int(len(value) * train_ratio)
        if random_seed is not None:
            value = value.sample(frac=1.0, random_state=random_seed)
        if train_data is None:
            train_data = value.head(end_index)
            val_data = value.tail(len(value) - end_index)
        else:
            train_data = train_data.append(value.head(end_index))
            val_data = val_data.append(value.tail(len(value) - end_index))
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    # endregion
    return train_data, val_data


def fix_csv(csv_dir):
    """
    修改师兄的bug,不必使用
    """
    for csv_name in os.listdir(csv_dir):
        if not csv_name.endswith('.csv'):
            continue
        csv_path = os.path.join(csv_dir, csv_name)
        if 'static' in csv_path and 'delete.csv' not in csv_name:  # 针对师兄做的数据集而采取的结果
            continue
        csv_info = pd.read_csv(csv_path, index_col=0)
        csv_info = csv_info.rename(columns={'    右肘c': '右肘c'})
        csv_info.to_csv(csv_path)


def get_csv_data_per_activity(csv_dir):
    """
    将所有的csv读取到all_csv_info中
    """
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
    return all_csv_info


save_dir = 'skeleton_pkl_data/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if __name__ == '__main__':
    csv_dir_list = [
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/摔倒",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/正常站立",

        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/back_against",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/climb",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/fall_behind",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/fall_toward",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/hand_out",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/head_and_hand_out",
        "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/2021摔倒数据集/normal_standing",

        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/back_against/back_against_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/climb/climb_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/fall_behind/fall_behind_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/fall_toward/fall_toward_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/hand_out/hand_out_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/head_and_hand_out/head_and_hand_out_static",
        # "/home/xzy/Data/扶梯项目数据集/姿态检测数据集/normal_standing_static_dynamic",

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

    pkl_info.to_pickle(f'{save_dir}all_skeleton_data.pkl')
    statistics_info = print_skeleton_info(pkl_info)
    # _ = pickle.load(open(f'{save_dir}all_skeleton_data.pkl', 'rb'))
    print("---save pkl successfully !!!---")
    print(f"num of {len(statistics_info.keys())} class data is : {pkl_info.shape[0]}")
    # endregion #####################################################
    # region 划分数据集
    train_data, val_data = get_train_val(info=pkl_info, train_ratio=0.8, random_seed=20)
    train_data.to_pickle(f'{save_dir}train_data.pkl')
    val_data.to_pickle(f'{save_dir}val_data.pkl')
    print(f"num of train data is : {train_data.shape[0]}")
    print(f"num of val data is : {val_data.shape[0]}")
    # endregion
