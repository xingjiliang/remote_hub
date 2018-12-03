import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
# tf.app.flags.DEFINE_integer('full_connection_layer_nums', 1, 'full connection layer nums')
# tf.app.flags.DEFINE_string('model_WTF', 'origin', 'zero')
# tf.app.flags.DEFINE_float('l2_lambda', 1e-5, 'l2 lambda')
# tf.app.flags.DEFINE_integer('day_grained_sequence_length', 7, 'day_grained_sequence_length')
# 除tf.app.run() main()函数可调用,在脚本调试中也应当可调用


class ModelSettings:
    def __init__(self):
        self.keep_prob = 0.5
        self.full_connection_layer_nums = 1
        self.model_WTF = "origin"
        self.l2_lambda = 1e-5
        self.day_grained_sequence_length = 7
        self.day_grained_cell_size = 50
        self.day_of_week_size = 7
        self.day_of_week_embedding_size = 5
        self.holidays_distance_size = 7*2+2
        self.holidays_distance_embedding_size = 5
        self.end_of_holidays_distance_size = 7+2
        self.end_of_holidays_distance_embedding_size = 5
        self.is_weekend_weekday_size = 2
        self.is_weekend_weekday_embedding_size = 5
        self.hour_grained_sequence_length = 24
        self.hour_grained_cell_size = 50
        self.hour_per_day_size = 24
        self.hour_per_day_embedding_size = 5


class Model:
    def __init__(self, is_training=False):
        settings = ModelSettings()
        self.actual_batch_size_scalar = tf.placeholder(dtype=tf.int32, shape=[1], name='actual_batch_size_scalar')
        actual_batch_size = self.actual_batch_size_scalar[0]

        with tf.variable_scope("day_grained_processing_frame"):
            self.day_of_week = tf.placeholder(dtype=tf.int32, shape=[None, settings.day_grained_sequence_length], name='day_of_week')
            self.holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, settings.day_grained_sequence_length], name='holidays_distance')
            self.end_of_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, settings.day_grained_sequence_length], name='end_of_holidays_distance')
            self.is_weekend_weekday = tf.placeholder(dtype=tf.int32, shape=[None, settings.day_grained_sequence_length], name='is_weekend_weekday')
            self.impression_per_day = tf.placeholder(dtype=tf.float64, shape=[None, settings.day_grained_sequence_length, 1], name='impression_per_day')

            self.day_of_week_embedding = tf.get_variable(name='day_of_week_embedding', shape=[settings.day_of_week_size, settings.day_of_week_embedding_size], dtype='float64')
            self.holidays_distance_embedding = tf.get_variable(name='holidays_distance_embedding', shape=[settings.holidays_distance_size, settings.holidays_distance_embedding_size], dtype='float64')
            self.end_of_holidays_distance_embedding = tf.get_variable(name='end_of_holidays_distance_embedding', shape=[settings.end_of_holidays_distance_size, settings.end_of_holidays_distance_embedding_size], dtype='float64')
            self.is_weekend_weekday_embedding = tf.get_variable(name='is_weekend_weekday_embedding', shape=[settings.is_weekend_weekday_size, settings.is_weekend_weekday_embedding_size], dtype='float64')
            day_grained_inputs = tf.concat(
                values=[
                tf.nn.embedding_lookup(self.day_of_week_embedding, self.day_of_week),
                tf.nn.embedding_lookup(self.holidays_distance_embedding, self.holidays_distance),
                tf.nn.embedding_lookup(self.end_of_holidays_distance_embedding, self.end_of_holidays_distance),
                tf.nn.embedding_lookup(self.is_weekend_weekday_embedding, self.is_weekend_weekday),
                self.impression_per_day
                ],
                axis=2
            )

            day_grained_forward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.day_grained_cell_size, use_peepholes=True, state_is_tuple=True)
            day_grained_backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.day_grained_cell_size, use_peepholes=True, state_is_tuple=True)
            if is_training and settings.keep_prob < 1:
                day_grained_forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(day_grained_forward_lstm_cell, input_keep_prob=settings.keep_prob)
                day_grained_backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(day_grained_backward_lstm_cell, input_keep_prob=settings.keep_prob)
            self.day_grained_initial_state_forward = day_grained_forward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            self.day_grained_initial_state_backward = day_grained_backward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            with tf.variable_scope('LSTM_LAYER'):
                day_grained_outputs, day_grained_outputs_state = tf.nn.bidirectional_dynamic_rnn(day_grained_forward_lstm_cell,
                                                                         day_grained_backward_lstm_cell,
                                                                         day_grained_inputs,
                                                                         sequence_length=self.day_grained_sequence_length,
                                                                         initial_state_fw=self.day_grained_initial_state_forward,
                                                                         initial_state_bw=self.day_grained_initial_state_backward)
            day_grained_output_forward = day_grained_outputs[0]
            day_grained_output_backward = day_grained_outputs[1]
            day_grained_output_h = tf.add(day_grained_output_forward, day_grained_output_backward)
            reduced_day_grained_output_h = tf.reduce_max(day_grained_output_h, 1)
            # day_grained_output_H = tf.concat([output_forward, output_backward],2)

            if is_training and settings.keep_prob < 1:
                reduced_day_grained_output_h = tf.nn.dropout(reduced_day_grained_output_h, keep_prob=settings.keep_prob)
            # LSTM,h=output_gate.*tanh(c)
            # day_grained_output_m = tf.tanh(reduced_day_grained_output_h)

        with tf.variable_scope("hour_grained_processing_frame"):
            self.hour_per_day = tf.placeholder(dtype=tf.int32, shape=[None, settings.hour_grained_sequence_length], name='hour_per_day')
            self.impression_per_hour = tf.placeholder(dtype=tf.float64, shape=[None, settings.hour_grained_sequence_length, 1], name='impression_per_hour')

            self.hour_per_day_embedding = tf.get_variable(name='hour_per_day_embedding', shape=[settings.hour_per_day_size, settings.hour_per_day_embedding_size], dtype='float64')
            hour_grained_inputs = tf.concat(
                values=[
                tf.nn.embedding_lookup(self.hour_per_day_embedding, self.hour_per_day),
                self.impression_per_hour
                ],
                axis=2
            )

            hour_grained_forward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.hour_grained_cell_size, use_peepholes=True, state_is_tuple=True)
            # hour_grained_backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.hour_grained_cell_size, use_peepholes=True, state_is_tuple=True)
            if is_training and settings.keep_prob < 1:
                hour_grained_forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(hour_grained_forward_lstm_cell, input_keep_prob=settings.keep_prob)
                # hour_grained_backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(hour_grained_backward_lstm_cell, input_keep_prob=settings.keep_prob)
            self.hour_grained_initial_state_forward = hour_grained_forward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            # self.hour_grained_initial_state_backward = hour_grained_backward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            with tf.variable_scope('LSTM_LAYER'):
                hour_grained_outputs, hour_grained_outputs_state = tf.nn.dynamic_rnn(hour_grained_forward_lstm_cell, hour_grained_inputs,
                                                                         sequence_length=self.hour_grained_sequence_length,
                                                                         initial_state=self.hour_grained_initial_state_forward)
                                                                         # initial_state_bw=self.hour_grained_initial_state_backward)
            # output_forward = outputs[0]
            # output_backward = outputs[1]
            # output_H = tf.add(output_forward, output_backward)
            # output_H = tf.concat([output_forward, output_backward],2)
            hour_grained_last_time_outputs = hour_grained_outputs[:, -1, :]
            if is_training and settings.keep_prob < 1:
                hour_grained_last_time_outputs = tf.nn.dropout(hour_grained_last_time_outputs, keep_prob=settings.keep_prob)

        self.prediction_day_day_of_week = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_day_of_week')
        self.prediction_day_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_holidays_distance')
        self.prediction_day_end_of_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_end_of_holidays_distance')
        self.prediction_day_is_weekend_weekday = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_is_weekend_weekday')

        # TODO:这里的全连接层没有隐层,无法刻画当天与先前数据的关系.
        prediction_day_inputs = tf.concat(
            values=[
                tf.nn.embedding_lookup(self.day_of_week_embedding, self.prediction_day_day_of_week).reshape(-1, settings.day_of_week_embedding_size),
                tf.nn.embedding_lookup(self.holidays_distance_embedding, self.prediction_day_holidays_distance).reshape(-1, settings.holidays_distance_embedding_size),
                tf.nn.embedding_lookup(self.end_of_holidays_distance_embedding, self.prediction_day_end_of_holidays_distance).reshape(-1, settings.end_of_holidays_distance_embedding_size),
                tf.nn.embedding_lookup(self.is_weekend_weekday_embedding, self.prediction_day_is_weekend_weekday).reshape(-1, settings.is_weekend_weekday_embedding_size)
            ],
            axis=1
        )

        batch_results = tf.concat([reduced_day_grained_output_h, hour_grained_last_time_outputs, prediction_day_inputs], 1)
        self.mlr_params = tf.get_variable(name="mlr_params", shape=[self.day_grained_cell_size + self.hour_grained_cell_size + settings.day_of_week_embedding_size + settings.holidays_distance_embedding_size + settings.end_of_holidays_distance_embedding_size + settings.is_weekend_weekday_embedding_size, 1], dtype='float64')
        self.mlr_bias_d = tf.get_variable(name="mlr_bias_d", shape=[1], dtype='float64')
        self._Y = tf.matmul(batch_results, self.mlr_params) + self.mlr_bias_d
        self.Y = tf.placeholder(name="true_value", shape=[None, 1])
        self.empirical_loss = tf.losses.mean_squared_error(self.Y, self._Y)
        # tf.get_collection('l2_loss')
        # for v in tf.trainable_variables():
        #     if v is word_embedding:
        #         continue
        #     tf.add_to_collection('l2_loss', v)
        # 这里的l2_lambda应由训练settings设置
        # self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(settings.l2_lambda),
        #                                                       weights_list=tf.get_collection('l2_loss'))
        # 后续会加入结构损失
        self.final_loss = self.empirical_loss
