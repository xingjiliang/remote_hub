import os
import numpy as np
import tensorflow as tf
import datetime
from sklearn import metrics

import config
import utils
from model_graphs import rnnV2 as model

FLAGS = tf.app.flags.FLAGS


def main(_):
    test_data_file_path = os.path.join(config.dataset_dir, "%d_test_data_%s_%s.npy" % (FLAGS.city_id, FLAGS.test_start_date, FLAGS.test_end_date))
    test_data = np.load(test_data_file_path)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope("model"):
                m = model.Model(is_training=False)
            saver = tf.train.Saver()
            if FLAGS.previous_model_epoch_times:
                saver.restore(sess,
                              os.path.join(config.model_params_dir, FLAGS.model_name, FLAGS.previous_model_epoch_times))
            else:
                saver.restore(sess,
                              os.path.join(config.model_params_dir, FLAGS.model_name, 'final'))
            model_mse_loss = None
            y = None
            _y = None
            goal = None
            start_time = datetime.datetime.now()
            for i in range(int(len(test_data) / float(FLAGS.test_data_batch_size)) + 1):
                test_data_batch = test_data[i * FLAGS.test_data_batch_size:(i + 1) * FLAGS.test_data_batch_size]
                feed_dict, goal_batch = utils.generate_feed_dict(test_data_batch, m, 1.0)
                model_mse_loss_batch, y_batch, _y_batch = sess.run([m.final_loss, m.Y, m._Y], feed_dict=feed_dict)
                model_mse_loss = np.concatenate([model_mse_loss, model_mse_loss_batch], 0) if model_mse_loss is not None else model_mse_loss_batch
                y = np.concatenate([y, y_batch], 0) if y is not None else y_batch
                _y = np.concatenate([_y, _y_batch], 0) if _y is not None else _y_batch
                goal = np.concatenate([goal, goal_batch], 0) if goal is not None else goal_batch
            end_time = datetime.datetime.now()
            y_true = goal[:, 3].reshape(-1, 1)
            y_pred = goal[:, 0].reshape(-1, 1) * _y + goal[:, 1].reshape(-1, 1)
            mse = metrics.mean_squared_error(y_true, y_pred)
            mae = metrics.mean_absolute_error(y_true, y_pred)
            mape = (abs(y_true - y_pred) / y_true).mean()
            print('在测试集%s上的MSE为%f,MAE为%f,MAPE为%f' % (test_data_file_path, mse, mae, mape))
            print('预测%d个样本总计运行%f秒' % (len(test_data), (end_time - start_time).total_seconds()))


if __name__ == "__main__":
    tf.app.run()
