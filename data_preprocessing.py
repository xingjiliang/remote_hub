# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import datetime

import config


def generate_data(tsv_file, save_as_file=True):
    city_id = int(tsv_file.split(sep='.')[0])
    origin_dataset_file_path = config.origin_dataset_dir_path + tsv_file
    holidays = config.holidays
    end_of_holidays = config.end_of_holidays
    weekend_weekdays = config.weekend_weekdays

    # 预处理小时粒度的数据
    df = pd.read_csv(origin_dataset_file_path, names=["city_id", "datetime", "count"], sep="\t")
    df['datetime'] = df.datetime.apply(lambda datetime: pd.to_datetime(datetime[0:13]))
    df = df[['datetime', 'count']].groupby('datetime').sum()
    df['datetime'] = df.index
    df['hour'] = df.datetime.apply(lambda datetime: datetime.hour)
    hour_grained_dataframe = df[["datetime", "hour", "count"]]
    # hour_grained_data = np.array(hour_grained_dataframe.values)
    # np.save(config.dataset_dir_path + str(city_id) + "_hour_grained_data.npy", hour_grained_data)

    # 预处理日粒度的数据
    df = pd.read_csv(origin_dataset_file_path, names=["city_id", "datetime", "count"], sep="\t")
    df['datetime'] = df.datetime.apply(lambda datetime: pd.to_datetime(datetime[0:10]))
    df = df[['datetime', 'count']].groupby('datetime').sum()
    df['datetime'] = df.index
    df['day_of_week'] = df.datetime.apply(lambda datetime: datetime.weekday())
    df['holidays_distance'] = 8
    df['end_of_holidays_distance'] = 8
    df['is_weekend_weekday'] = 0
    for i in range(df.shape[0]):
        for holiday in holidays:
            tmp = (pd.to_datetime(holiday) - df.iloc[i, 1]).days
            if abs(tmp) < abs(df.iloc[i, 3]):
                df.iloc[i, 3] = tmp
        for end_of_holiday in end_of_holidays:
            tmp = (pd.to_datetime(end_of_holiday) - df.iloc[i, 1]).days
            if tmp < df.iloc[i, 4] and tmp >= 0:
                df.iloc[i, 4] = tmp
        for weekend_weekday in weekend_weekdays:
            if pd.to_datetime(weekend_weekday) == df.iloc[i,1]:
                df.iloc[i, 5] = 1
    df.holidays_distance = df.holidays_distance + 7
    day_grained_dataframe = df[['datetime', 'day_of_week', 'holidays_distance', 'end_of_holidays_distance', 'is_weekend_weekday', 'count']]
    # day_grained_data = np.array(day_grained_dataframe.values)
    # np.save(config.dataset_dir_path + str(city_id) + "_day_grained_data.npy", day_grained_data)

    # 将数据处理为可输入模型的格式
    may_be_predict_date = pd.to_datetime(config.dateset_start_date) + datetime.timedelta(days=7)
    data = []
    for i in range(len(hour_grained_dataframe)):
        if hour_grained_dataframe.iloc[i].datetime < may_be_predict_date:
            continue
        else:
            day_grained_8_days_data = []
            for delta_days in range(7, 0, -1):
                day_grained_8_days_data.append(day_grained_dataframe.loc[(hour_grained_dataframe.iloc[i].datetime - datetime.timedelta(days=delta_days)).strftime('%Y-%m-%d')].values[1:])

            day_grained_8_days_data.append(day_grained_dataframe.loc[hour_grained_dataframe.iloc[i].datetime.strftime('%Y-%m-%d')].values[1:])
            day_grained_8_days_data = np.array(day_grained_8_days_data, dtype='float64')
            day_growth_rate_vector = (np.array(day_grained_8_days_data[:, 4], dtype='float64') / float(day_grained_8_days_data[:, 4][0]) - 1).reshape(-1, 1)
            day_grained_8_days_data = np.concatenate([day_grained_8_days_data, day_growth_rate_vector], 1)
            hour_grained_data = np.array(hour_grained_dataframe.iloc[i - 25: i].values[:, 1:], dtype='float64')
            hour_growth_rate_vector = (hour_grained_data[:, 1] / hour_grained_data[0, 1] - 1).reshape(-1, 1)
            data.append([hour_grained_dataframe.iloc[i].values[0],
                        day_grained_8_days_data[:-1],
                        day_grained_8_days_data[-1, :-1],
                        np.concatenate([hour_grained_data, hour_growth_rate_vector], 1)[1:],
                        day_grained_8_days_data[-1, -1]]
                        )
    data = np.array(data)
    if save_as_file:
        np.save(config.dataset_dir_path + str(city_id) + "_data.npy", data)
    return data


def generate_train_and_test_data(
        data,
        train_data_start_date,
        train_data_end_date,
        test_data_start_date,
        test_data_end_date,
        save_as_file=True,
        city_id=0
):
    train_data_start_index = np.argwhere(data[:, 0] == pd.to_datetime(train_data_start_date))[0, 0]
    train_data_end_index = np.argwhere(data[:, 0] == pd.to_datetime(train_data_end_date))[0, 0]
    test_data_start_index = np.argwhere(data[:, 0] == pd.to_datetime(test_data_start_date))[0, 0]
    test_data_end_index = np.argwhere(data[:, 0] == pd.to_datetime(test_data_end_date))[0, 0]
    train_data = data[train_data_start_index:train_data_end_index]
    test_data = data[test_data_start_index:test_data_end_index]
    if save_as_file:
        np.save(config.dataset_dir_path + str(city_id) + "_train_data_%s_%s.npy" % (train_data_start_date, train_data_end_date), train_data)
        np.save(config.dataset_dir_path + str(city_id) + "_test_data_%s_%s.npy" % (test_data_start_date, test_data_end_date), test_data)
    return train_data, test_data


if __name__ == '__main__':
    generate_train_and_test_data(data=generate_data("1.tsv"),
                                 train_data_start_date='2018-03-19',
                                 train_data_end_date='2018-08-16',
                                 test_data_start_date='2018-08-16',
                                 test_data_end_date='2018-09-21',
                                 city_id=1)
