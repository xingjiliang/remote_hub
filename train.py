import numpy as np
import tensorflow as tf

import config
import data_manager;

class TrainSettings:
    def __init__(
            self,
            # 未来可能会加入多个城市的数据
            city="bj",
            batch_size=64,
            num_epochs=100
    ):
        dataset_path = config.dataset_dir_path
        self.hoursize_train_data_file_path = dataset_path + "hoursize_train_data.npy"
        self.daysize_train_data_file_path = dataset_path + "daysize_train_data.npy"
        self.batch_size = batch_size
        self.num_epochs = num_epochs

def main(_):
    train_settings = TrainSettings()
    train_data = data_manager.load_data(train_settings.hoursize_train_data_file_path)


if __name__ == "__main__":
	tf.app.run()