import pandas as pd
import numpy as np
import config

if __name__ == "__main__":
    origin_dataset_file_path = config.origin_dataset_dir_path + config.dataset_file_name
    holidays = config.holidays
    end_of_holidays = config.end_of_holidays
    weekend_weekdays = config.weekend_weekdays

    df = pd.read_csv(origin_dataset_file_path, names=["cityid", "datetime", "count"], sep="\t")
    # 预处理小时粒度的数据
    df['datetime'] = df.datetime.apply(lambda datetime:pd.to_datetime(datetime[0:13]))
    df = df[['datetime', 'count']].groupby('datetime').sum()
    df['datetime'] = df.index
    df['hour'] = df.datetime.apply(lambda datetime: datetime.hour)
    hoursize_data = np.array(df[["datetime", "hour", "count"]].values)
    np.save(config.dataset_dir_path + "hoursize_data.npy", hoursize_data)

    df = pd.read_csv(origin_dataset_file_path, names=["cityid", "datetime", "count"], sep="\t")
    df['datetime'] = df.datetime.apply(lambda datetime: pd.to_datetime(datetime[0:10]))
    df = df[['datetime', 'count']].groupby('datetime').sum()
    df['datetime'] = df.index
    df['day_of_week'] = df.datetime.apply(lambda datetime:datetime.weekday())
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
    daysize_data = np.array(df[['datetime', 'day_of_week', 'holidays_distance', 'end_of_holidays_distance', 'is_weekend_weekday', 'count']].values)
    np.save(config.dataset_dir_path + "daysize_data.npy", daysize_data)
