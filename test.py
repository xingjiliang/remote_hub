import numpy as np
import tensorflow as tf

import config

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('test_start_date', '2018-08-16', 'test start date')
tf.app.flags.DEFINE_string('test_end_date', '2018-09-21', 'test end date')


class TestSettings:
    def __init__(
            self,
            city_id,
            test_start_date,
            test_end_date,
    ):
        self.city_id = city_id
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.test_data_file_path = config.dataset_dir_path + str(city_id) + "_test_data_%s_%s.npy" % (test_start_date, test_end_date)


def load_test_data(city_id=FLAGS.city_id,
                   test_start_date=FLAGS.test_start_date,
                   test_end_date=FLAGS.train_end_date):
    test_settings = TestSettings()
    test_data = np.load(test_settings.test_data_file_path)
    return test_data


