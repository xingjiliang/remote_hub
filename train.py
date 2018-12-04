import numpy as np
import tensorflow as tf
import datetime

import config
import utils
from model_graphs import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('city_id', 1, 'city id')
tf.app.flags.DEFINE_string('train_start_date', '2018-03-19', 'train start date')
tf.app.flags.DEFINE_string('train_end_date', '2018-08-16', 'train end date')
# tf.app.flags.DEFINE_float('l2_lambda', 1e-5, 'l2 lambda')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'epoch times')
tf.app.flags.DEFINE_bool('test_when_training', True, 'test when training')
tf.app.flags.DEFINE_float('learning_rate', 1000000, 'learning_rate')


class TrainSettings:
    def __init__(
            self,
            city_id,
            train_start_date,
            train_end_date,
            batch_size,
            num_epochs,
            keep_prob
    ):
        self.city_id = city_id
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.train_data_file_path = config.dataset_dir_path + str(city_id) + "_train_data_%s_%s.npy" % (train_start_date, train_end_date)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.keep_prob = keep_prob


def main(args):
    train_settings = TrainSettings(city_id=FLAGS.city_id,
                                   train_start_date=FLAGS.train_start_date,
                                   train_end_date=FLAGS.train_end_date,
                                   batch_size=FLAGS.batch_size,
                                   num_epochs=FLAGS.num_epochs,
                                   keep_prob=FLAGS.keep_prob)
    train_data = np.load(train_settings.train_data_file_path)
    # if FLAGS.test_when_training:
    #     import test
    #     test_data = test.load_test_data(FLAGS.city_id, FLAGS.test_start_date, FLAGS.test_end_date)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = model.Model(is_training=True)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
            optimizer_term = optimizer.minimize(m.empirical_loss, global_step=global_step)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=None)

            # if FLAGS.test_when_training:

                # pass
                # todo:产生test_feed_dict

            minimum_MSE_loss = 1e10
            for epoch in range(train_settings.num_epochs):
                random_order = list(range(len(train_data)))
                np.random.shuffle(random_order)
                for i in range(int(len(random_order) / float(train_settings.batch_size)) + 1):
                    day_of_week_batch = []
                    holidays_distance_batch = []
                    end_of_holidays_distance_batch = []
                    is_weekend_weekday_batch = []
                    impression_per_day_batch = []
                    prediction_day_day_of_week_batch = []
                    prediction_day_holidays_distance_batch = []
                    prediction_day_end_of_holidays_distance_batch = []
                    prediction_day_is_weekend_weekday_batch = []
                    hour_per_day_batch = []
                    impression_per_hour_batch = []
                    Y_batch = []
                    input_batch_numbers = random_order[i * train_settings.batch_size:(i + 1) * train_settings.batch_size]
                    for k in input_batch_numbers:
                        day_of_week_batch.append(train_data[k][1][:, 0])
                        holidays_distance_batch.append(train_data[k][1][:, 1])
                        end_of_holidays_distance_batch.append(train_data[k][1][:, 2])
                        is_weekend_weekday_batch.append(train_data[k][1][:, 3])
                        impression_per_day_batch.append(train_data[k][1][:, 4])
                        prediction_day_day_of_week_batch.append(train_data[k][2][0])
                        prediction_day_holidays_distance_batch.append(train_data[k][2][1])
                        prediction_day_end_of_holidays_distance_batch.append(train_data[k][2][2])
                        prediction_day_is_weekend_weekday_batch .append(train_data[k][2][3])
                        hour_per_day_batch.append(train_data[k][3][:, 0])
                        impression_per_hour_batch.append(train_data[k][3][:, 1])
                        Y_batch.append(train_data[k][4])
                    day_of_week_batch = np.array(day_of_week_batch, dtype='int32')
                    holidays_distance_batch = np.array(holidays_distance_batch, dtype='int32')
                    end_of_holidays_distance_batch = np.array(end_of_holidays_distance_batch, dtype='int32')
                    is_weekend_weekday_batch = np.array(is_weekend_weekday_batch, dtype='int32')
                    impression_per_day_batch = np.array(impression_per_day_batch, dtype='float64').reshape([-1, 7, 1])
                    prediction_day_day_of_week_batch = np.array(prediction_day_day_of_week_batch, dtype='int32').reshape([-1, 1])
                    prediction_day_holidays_distance_batch = np.array(prediction_day_holidays_distance_batch, dtype='int32').reshape([-1, 1])
                    prediction_day_end_of_holidays_distance_batch = np.array(prediction_day_end_of_holidays_distance_batch, dtype='int32').reshape([-1, 1])
                    prediction_day_is_weekend_weekday_batch = np.array(prediction_day_is_weekend_weekday_batch, dtype='int32').reshape([-1, 1])
                    hour_per_day_batch = np.array(hour_per_day_batch, dtype='int32')
                    impression_per_hour_batch = np.array(impression_per_hour_batch, dtype='float64').reshape([-1, 24, 1])
                    Y_batch = np.array(Y_batch, dtype='float64').reshape([-1, 1])

                    feed_dict = utils.generate_feed_dict(m,
                                                         day_of_week_batch,
                                                         holidays_distance_batch,
                                                         end_of_holidays_distance_batch,
                                                         is_weekend_weekday_batch,
                                                         impression_per_day_batch,
                                                         prediction_day_day_of_week_batch,
                                                         prediction_day_holidays_distance_batch,
                                                         prediction_day_end_of_holidays_distance_batch,
                                                         prediction_day_is_weekend_weekday_batch,
                                                         hour_per_day_batch,
                                                         impression_per_hour_batch,
                                                         Y_batch,
                                                         Y_batch.shape[0],
                                                         train_settings.keep_prob)
                    temp, step, final_loss, y, _y = sess.run([optimizer_term,
                                                              global_step,
                                                              m.final_loss,
                                                              m.Y,
                                                              m._Y],
                                                              feed_dict=feed_dict)
                    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if step % 100 == 0:
                        info = "[{}] epoch {}, final_loss {:g}.".format(time_string, epoch, final_loss)
                        print(info)
                        print(np.concatenate([y.reshape(-1, 1), _y.reshape(-1, 1)], 1))
                    current_step = tf.train.global_step(sess, global_step)
                if epoch > 10:
                    # MSE_loss_on_test_data =
                    # if epoch % 10 == 0 or MSE_loss_on_test_data < minimum_MSE_loss:
                    if epoch % 10 == 0:
                        print('The current model is being stored.')
                        path = saver.save(sess, config.model_path + 'RegressionModel', global_step=current_step)
                        info = 'The current model has been stored to ' + path
                        print(info)


if __name__ == "__main__":
    tf.app.run()
