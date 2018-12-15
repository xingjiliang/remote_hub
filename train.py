import numpy as np
import tensorflow as tf
import datetime

import config
import utils
from model_graphs import modelV2 as model

FLAGS = config.FLAGS


def main(_):
    train_data_file_path = config.dataset_dir_path + str(FLAGS.city_id) + "_train_data_%s_%s.npy" % (FLAGS.train_start_date, FLAGS.train_end_date)
    train_data = np.load(train_data_file_path)
    # if FLAGS.test_when_training:
    #     import test
    #     test_data = test.load_test_data(FLAGS.city_id, FLAGS.test_start_date, FLAGS.test_end_date)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model_settings = model.ModelSettings(
                    keep_prob=FLAGS.keep_prob,
                    l2_lambda=FLAGS.l2_lambda,
                    day_grained_sequence_length=FLAGS.day_grained_sequence_length,
                    day_grained_cell_size=FLAGS.day_grained_cell_size,
                    day_of_week_embedding_size=FLAGS.day_of_week_embedding_size,
                    holidays_distance_size=FLAGS.holidays_distance_size,
                    holidays_distance_embedding_size=FLAGS.holidays_distance_embedding_size,
                    end_of_holidays_distance_size=FLAGS.end_of_holidays_distance_size,
                    end_of_holidays_distance_embedding_size=FLAGS.end_of_holidays_distance_embedding_size,
                    is_weekend_weekday_embedding_size=FLAGS.is_weekend_weekday_embedding_size,
                    hour_grained_sequence_length=FLAGS.hour_grained_sequence_length,
                    hour_grained_cell_size=FLAGS.hour_grained_cell_size,
                    hour_per_day_embedding_size=FLAGS.hour_per_day_embedding_size,
                    fcn_layer_nums=FLAGS.fcn_layer_nums,
                    fcn_hidden_layer_size=FLAGS.fcn_hidden_layer_size
                )
                m = model.Model(model_settings=model_settings, is_training=True)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
            optimizer_term = optimizer.minimize(m.empirical_loss, global_step=global_step)
            saver = tf.train.Saver(max_to_keep=None)
            if len(FLAGS.previous_model_name) and FLAGS.previous_model_epoch_times:
                saver.restore(sess, "%s%s_%d_epochs" % (
                config.model_path, FLAGS.previous_model_name, FLAGS.previous_model_epoch_times))
            else:
                sess.run(tf.global_variables_initializer())

            # if FLAGS.test_when_training:

                # pass
                # todo:产生test_feed_dict

            start_epoch_num = 1
            end_epoch_num = FLAGS.num_epochs+1
            if FLAGS.load_previous_model:
                start_epoch_num += FLAGS.previous_model_epoch_times
                end_epoch_num += FLAGS.previous_model_epoch_times
            for epoch in range(start_epoch_num, end_epoch_num):
                random_order = list(range(len(train_data)))
                np.random.shuffle(random_order)
                for i in range(int(len(random_order) / float(FLAGS.batch_size)) + 1):
                    input_batch_indexes = random_order[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
                    train_data_batch = train_data.take(input_batch_indexes, axis=0)
                    feed_dict, true_impression_batch = utils.generate_feed_dict(train_data_batch, m, FLAGS.keep_prob)
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
                        print(np.concatenate([(y.reshape(-1, 1) + 1) * true_impression_batch, (_y.reshape(-1, 1) + 1) * true_impression_batch], 1))
                    current_step = tf.train.global_step(sess, global_step)
                if epoch > 10:
                    # MSE_loss_on_test_data =
                    # if epoch % 10 == 0 or MSE_loss_on_test_data < minimum_MSE_loss:
                    if epoch % 10 == 0:
                        print('The current model is being stored.')
                        path = saver.save(sess, config.model_path + 'RegressionModel_%d_epochs' % epoch)
                        info = 'The current model has been stored to ' + path
                        print(info)


if __name__ == "__main__":
    tf.app.run()
