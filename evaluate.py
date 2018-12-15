import numpy as np
import tensorflow as tf

import config
from model_graphs import modelV2 as model

FLAGS = tf.app.flags.FLAGS


def main(_):
    test_data_file_path = config.dataset_dir_path + str(FLAGS.city_id) + "_test_data_%s_%s.npy" % (FLAGS.test_start_date, FLAGS.test_end_date)
    test_data = np.load(test_data_file_path)
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            with tf.variable_scope("model"):
                model = model.Model(is_training=False, word_embeddings=wordembedding, settings=test_settings)


if __name__ == "__main__":
    tf.app.run()
