import numpy as np


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
    else:
        data = data_or_data_file_path




    train_data =
    test_data =
    return (train_data,test_data)
