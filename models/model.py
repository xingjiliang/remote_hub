import tensorflow as tf
import numpy as np

class ModelSettings:
    def __init__(self):
        self.keep_prob = 0.5
        self.num_layers = 1
        #hour_size,day_size表示相应的RNN的序列长度,而非一天,一周有多少小时,多少天
        self.hour_size = 24
        self.day_size = 7
        self.cell_size = 50
        self.hour_grained_cell_size = 50
        self.keep_prob = 0.5
        self.num_layers = 1
        self.model_WTF = "origin"
        self.day_of_week_size = 7
        self.day_of_week_embedding_size = 5
        self.holidays_distance_size = 7*2+2
        self.holidays_distance_embedding_size = 5
        self.end_of_holidays_distance_size = 7+2
        self.end_of_holidays_distance_embedding_size = 5
        self.is_weekend_weekday_size = 2
        self.is_weekend_weekday_embedding_size = 5
        self.hour_per_day_size = 24
        self.hour_per_day_embedding_size = 5

class Model:
    def __init__(self, is_training=False):
        settings = ModelSettings()

        with tf.variable_scope("day_grained_processing_frame"):
            self.day_of_week = tf.placeholder(dtype=tf.int8, shape=[None, settings.day_size], name='day_of_week')
            self.holidays_distance = tf.placeholder(dtype=tf.int8, shape=[None, settings.day_size], name='holidays_distance')
            self.end_of_holidays_distance = tf.placeholder(dtype=tf.int8, shape=[None, settings.day_size], name='end_of_holidays_distance')
            self.is_weekend_weekday = tf.placeholder(dtype=tf.int8, shape=[None, settings.day_size], name='is_weekend_weekday')
            self.impression_per_day = tf.placeholder(dtype=tf.int64, shape=[None, settings.day_size], name='impression_per_day')

            self.day_of_week_embedding = tf.get_variable(name='day_of_week_embedding', shape=[settings.day_of_week_size, settings.day_of_week_embedding_size])
            self.holidays_distance_embedding = tf.get_variable(name='holidays_distance_embedding', shape=[settings.holidays_distance_size, settings.holidays_distance_embedding_size])
            self.end_of_holidays_distance_embedding = tf.get_variable(name='end_of_holidays_distance_embedding', shape=[settings.end_of_holidays_distance_size, settings.end_of_holidays_distance_embedding_size])
            self.is_weekend_weekday_embedding = tf.get_variable(name='is_weekend_weekday_embedding', shape=[settings.is_weekend_weekday_size, settings.is_weekend_weekday_embedding_size])
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

            forward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.cell_size, use_peepholes=True, state_is_tuple=True)
            backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(settings.cell_size, use_peepholes=True, state_is_tuple=True)
            if is_training and settings.keep_prob < 1:
                forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(forward_lstm_cell, input_keep_prob=settings.keep_prob)
                backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(backward_lstm_cell, input_keep_prob=settings.keep_prob)
            self._initial_state_forward = forward_lstm_cell.zero_state(settings.batch_size, tf.float32)
            self._initial_state_backward = backward_lstm_cell.zero_state(settings.batch_size, tf.float32)
            with tf.variable_scope('LSTM_LAYER'):
                outputs, outputs_state = tf.nn.bidirectional_dynamic_rnn(forward_lstm_cell, backward_lstm_cell, day_grained_inputs,
                                                                         sequence_length=self.day_size,
                                                                         initial_state_fw=self._initial_state_forward,
                                                                         initial_state_bw=self._initial_state_backward)
            output_forward = outputs[0]
            output_backward = outputs[1]
            output_H = tf.add(output_forward, output_backward)
            # output_H = tf.concat([output_forward, output_backward],2)

            if is_training and settings.keep_prob < 1:
                output_H = tf.nn.dropout(output_H, keep_prob=settings.keep_prob)


        with tf.variable_scope("hour_grained_processing_frame"):
            self.hour_per_day = tf.placeholder(dtype=tf.int8, shape=[None, settings.hour_size], name='hour')
            self.impression_per_hour = tf.placeholder(dtype=tf.int64, shape=[None, settings.hour_size], name='impression_per_hour')
            self.hour_per_day_embedding = tf.get_variable(name='hour_per_day_embedding', shape=[settings.hour_per_day_size, settings.hour_per_day_embedding_size])
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
            self.hour_grained_initial_state_forward = hour_grained_forward_lstm_cell.zero_state(settings.batch_size, tf.float32)
            # self.hour_grained_initial_state_backward = hour_grained_backward_lstm_cell.zero_state(settings.batch_size, tf.float32)
            with tf.variable_scope('LSTM_LAYER'):
                outputs, outputs_state = tf.nn.dynamic_rnn(hour_grained_forward_lstm_cell, hour_grained_inputs,
                                                                         sequence_length=self.hour_size,
                                                                         initial_state=self.hour_grained_initial_state_forward)
                                                                         # initial_state_bw=self.hour_grained_initial_state_backward)
            # output_forward = outputs[0]
            # output_backward = outputs[1]
            # output_H = tf.add(output_forward, output_backward)
            # output_H = tf.concat([output_forward, output_backward],2)
