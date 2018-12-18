import os
import importlib
import numpy as np
import tensorflow as tf
import datetime

import config
import utils

FLAGS = config.FLAGS


def main(_):
    model = importlib.import_module(config.model_graph_dir + '.' + FLAGS.model_name)
    train_data_file_path = os.path.join(config.dataset_dir, "%d_train_data_%s_%s.npy" % (FLAGS.city_id, FLAGS.train_start_date, FLAGS.train_end_date))
    train_data = np.load(train_data_file_path)
    # if FLAGS.evaluate_when_training:
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
            saver = tf.train.Saver(max_to_keep=None)
            if FLAGS.previous_model_epoch_times:
                saver.restore(sess,
                              os.path.join(config.model_params_dir, FLAGS.model_name, str(FLAGS.previous_model_epoch_times)))
            else:
                sess.run(tf.global_variables_initializer())

            # if FLAGS.evaluate_when_training:

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
                    feed_dict, goal_batch = utils.generate_feed_dict(train_data_batch, m, FLAGS.keep_prob)
                    temp, step, final_loss, y, _y = sess.run([optimizer_term,
                                                              global_step,
                                                              m.final_loss,
                                                              m.Y,
                                                              m._Y],
                                                             feed_dict=feed_dict)
                    time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if step % 100 == 0:
                        info = "{} - epoch {}, final_loss {:g}.".format(time_string, epoch, final_loss)
                        print(info)
                        print(np.concatenate([goal_batch[:, 3].reshape(-1, 1), (_y.reshape(-1, 1) + 1) * goal_batch[:, 0].reshape(-1, 1)], 1))
                    current_step = tf.train.global_step(sess, global_step)
                if epoch > 10:
                    # MSE_loss_on_test_data =
                    # if epoch % 10 == 0 or MSE_loss_on_test_data < minimum_MSE_loss:
                    if epoch % 10 == 0:
                        time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print('%s - The current model is being stored.' % time_string)
                        path = saver.save(sess, os.path.join(config.model_params_dir, FLAGS.model_name, str(epoch)))
                        print('%s - The current model has been stored to ' % time_string + path)
            path = saver.save(sess, os.path.join(config.model_params_dir, FLAGS.model_name, 'final'))
            time_string = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print('%s - The lasted model has been stored to ' % time_string + path)


if __name__ == "__main__":
    tf.app.run()
