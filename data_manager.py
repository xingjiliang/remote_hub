import numpy as np
import datetime
import pandas as pd


def generate_train_and_test_data(
        data_or_data_file_path,
        start_date,
        end_date,
        train_data_start_date,
        train_data_end_date,
        test_data_start_date,
        test_data_end_date
):
    if isinstance(data_or_data_file_path, str):
        data = np.load(data_or_data_file_path)
        day_grained_data = np.load()
        hour_grained_data = np.load()
    else:
        data = data_or_data_file_path

    train_data_may_be_trained_datetime = pd.to_datetime(train_data_start_date) + datetime.timedelta(days=7)
    for i in range(len(hour_grained_data)):
        if hour_grained_data[i][0] < train_data_may_be_trained_datetime:
            continue
        else:
            record =





    train_data =
    test_data =
    return (train_data,test_data)
