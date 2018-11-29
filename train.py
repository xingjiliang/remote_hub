import numpy as np
import tensorflow as tf

import config
import data_manager


class TrainSettings:
    def __init__(
            self,
            # 未来可能会加入多个城市的数据
            city_id=1,
            l2_lambda=1e-5,
            batch_size=64,
            num_epochs=100,
    ):
        dataset_path = config.dataset_dir_path
        #因为样本总量较小,为每一个小时粒度生成一个样本.
        self.train_data_file_path = dataset_path + str(city_id) + "_train_data.npy"
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.num_epochs = num_epochs


def main():
    train_settings = TrainSettings()
    day_grained_train_data = np.load(train_settings.day_grained_train_data_file_path)
    hour_grained_train_data = np.load(train_settings.hour_grained_train_data_file_path)


if __name__ == "__main__":
    tf.app.run()
