import tensorflow as tf


# 前提，当修改了一个模型的结构，之前模型的参数就没用了
# 最科学的做法是将ModelSettings改为ModelConfig，即一个Model有其固定的结构
class ModelSettings:
    day_grained_sequence_length = 7
    day_grained_cell_size = 30
    day_of_week_size = 7
    day_of_week_embedding_size = 5
    holidays_distance_size = 7*2+2
    holidays_distance_embedding_size = 5
    end_of_holidays_distance_size = 7+2
    end_of_holidays_distance_embedding_size = 5
    is_weekend_weekday_size = 2
    is_weekend_weekday_embedding_size = 5
    hour_grained_sequence_length = 24
    hour_grained_cell_size = 15
    hour_per_day_size = 24
    hour_per_day_embedding_size = 5
    fcn_layer_nums = 1
    fcn_hidden_layer_size = 20


class Model:
    def __init__(self, is_training=False):
        model_settings = ModelSettings()
        self.keep_prob = tf.placeholder(dtype=tf.float64, name='keep_prob')
        self.actual_batch_size_scalar = tf.placeholder(dtype=tf.int32, shape=[1], name='actual_batch_size_scalar')
        actual_batch_size = self.actual_batch_size_scalar[0]

        with tf.variable_scope("day_grained_processing_frame"):
            self.day_of_week = tf.placeholder(dtype=tf.int32, shape=[None, model_settings.day_grained_sequence_length], name='day_of_week')
            self.holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, model_settings.day_grained_sequence_length], name='holidays_distance')
            self.end_of_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, model_settings.day_grained_sequence_length], name='end_of_holidays_distance')
            self.is_weekend_weekday = tf.placeholder(dtype=tf.int32, shape=[None, model_settings.day_grained_sequence_length], name='is_weekend_weekday')
            self.impression_per_day = tf.placeholder(dtype=tf.float64, shape=[None, model_settings.day_grained_sequence_length, 1], name='impression_per_day')

            self.day_of_week_embedding = tf.get_variable(name='day_of_week_embedding', shape=[model_settings.day_of_week_size, model_settings.day_of_week_embedding_size], dtype='float64')
            self.holidays_distance_embedding = tf.get_variable(name='holidays_distance_embedding', shape=[model_settings.holidays_distance_size, model_settings.holidays_distance_embedding_size], dtype='float64')
            self.end_of_holidays_distance_embedding = tf.get_variable(name='end_of_holidays_distance_embedding', shape=[model_settings.end_of_holidays_distance_size, model_settings.end_of_holidays_distance_embedding_size], dtype='float64')
            self.is_weekend_weekday_embedding = tf.get_variable(name='is_weekend_weekday_embedding', shape=[model_settings.is_weekend_weekday_size, model_settings.is_weekend_weekday_embedding_size], dtype='float64')
            self.day_grained_inputs = tf.concat(
                values=[
                tf.nn.embedding_lookup(self.day_of_week_embedding, self.day_of_week),
                tf.nn.embedding_lookup(self.holidays_distance_embedding, self.holidays_distance),
                tf.nn.embedding_lookup(self.end_of_holidays_distance_embedding, self.end_of_holidays_distance),
                tf.nn.embedding_lookup(self.is_weekend_weekday_embedding, self.is_weekend_weekday),
                self.impression_per_day
                ],
                axis=2
            )

            day_grained_forward_lstm_cell = tf.nn.rnn_cell.LSTMCell(model_settings.day_grained_cell_size, use_peepholes=False, state_is_tuple=True)
            day_grained_backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(model_settings.day_grained_cell_size, use_peepholes=False, state_is_tuple=True)
            if is_training:
                day_grained_forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(day_grained_forward_lstm_cell, input_keep_prob=self.keep_prob)
                day_grained_backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(day_grained_backward_lstm_cell, input_keep_prob=self.keep_prob)
            day_grained_initial_state_forward = day_grained_forward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            day_grained_initial_state_backward = day_grained_backward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            with tf.variable_scope('LSTM_LAYER'):
                self.day_grained_outputs, self.day_grained_outputs_state = tf.nn.bidirectional_dynamic_rnn(day_grained_forward_lstm_cell,
                                                                                                           day_grained_backward_lstm_cell,
                                                                                                           self.day_grained_inputs,
                                                                                                           initial_state_fw=day_grained_initial_state_forward,
                                                                                                           initial_state_bw=day_grained_initial_state_backward)
            self.day_grained_output_forward = self.day_grained_outputs[0]
            self.day_grained_output_backward = self.day_grained_outputs[1]
            self.day_grained_output_h = tf.concat([self.day_grained_output_forward, self.day_grained_output_backward], 2)
            self.reduced_day_grained_output_h = tf.reduce_max(self.day_grained_output_h, 1)
            # day_grained_output_H = tf.concat([output_forward, output_backward],2)

            if is_training:
                self.reduced_day_grained_output_h = tf.nn.dropout(self.reduced_day_grained_output_h, keep_prob=self.keep_prob)
            # LSTM,h=output_gate.*tanh(c)
            # self.day_grained_output_m = tf.tanh(self.reduced_day_grained_output_h)

        with tf.variable_scope("hour_grained_processing_frame"):
            self.hour_per_day = tf.placeholder(dtype=tf.int32, shape=[None, model_settings.hour_grained_sequence_length], name='hour_per_day')
            self.impression_per_hour = tf.placeholder(dtype=tf.float64, shape=[None, model_settings.hour_grained_sequence_length, 1], name='impression_per_hour')

            self.hour_per_day_embedding = tf.get_variable(name='hour_per_day_embedding', shape=[model_settings.hour_per_day_size, model_settings.hour_per_day_embedding_size], dtype='float64')
            self.hour_grained_inputs = tf.concat(
                values=[
                tf.nn.embedding_lookup(self.hour_per_day_embedding, self.hour_per_day),
                self.impression_per_hour
                ],
                axis=2
            )

            hour_grained_forward_lstm_cell = tf.nn.rnn_cell.LSTMCell(model_settings.hour_grained_cell_size, use_peepholes=False, state_is_tuple=True)
            # hour_grained_backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(model_settings.hour_grained_cell_size, use_peepholes=True, state_is_tuple=True)
            if is_training:
                hour_grained_forward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(hour_grained_forward_lstm_cell, input_keep_prob=self.keep_prob)
                # hour_grained_backward_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(hour_grained_backward_lstm_cell, input_keep_prob=model_settings.keep_prob)
            hour_grained_initial_state_forward = hour_grained_forward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            # self.hour_grained_initial_state_backward = hour_grained_backward_lstm_cell.zero_state(actual_batch_size, tf.float64)
            with tf.variable_scope('LSTM_LAYER'):
                self.hour_grained_outputs, self.hour_grained_outputs_state = tf.nn.dynamic_rnn(hour_grained_forward_lstm_cell,
                                                                                     self.hour_grained_inputs,
                                                                                     initial_state=hour_grained_initial_state_forward)
                                                                                     # initial_state_bw=self.hour_grained_initial_state_backward)
            # output_forward = outputs[0]
            # output_backward = outputs[1]
            # output_H = tf.add(output_forward, output_backward)
            # output_H = tf.concat([output_forward, output_backward],2)
            self.hour_grained_last_time_outputs = self.hour_grained_outputs[:, -1, :]
            if is_training:
                self.hour_grained_last_time_outputs = tf.nn.dropout(self.hour_grained_last_time_outputs, keep_prob=self.keep_prob)

        self.prediction_day_day_of_week = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_day_of_week')
        self.prediction_day_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_holidays_distance')
        self.prediction_day_end_of_holidays_distance = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_end_of_holidays_distance')
        self.prediction_day_is_weekend_weekday = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='prediction_day_is_weekend_weekday')

        self.prediction_day_inputs = tf.concat(
            values=[
                tf.reshape(tf.nn.embedding_lookup(self.day_of_week_embedding, self.prediction_day_day_of_week), [-1, model_settings.day_of_week_embedding_size]),
                tf.reshape(tf.nn.embedding_lookup(self.holidays_distance_embedding, self.prediction_day_holidays_distance), [-1, model_settings.holidays_distance_embedding_size]),
                tf.reshape(tf.nn.embedding_lookup(self.end_of_holidays_distance_embedding, self.prediction_day_end_of_holidays_distance), [-1, model_settings.end_of_holidays_distance_embedding_size]),
                tf.reshape(tf.nn.embedding_lookup(self.is_weekend_weekday_embedding, self.prediction_day_is_weekend_weekday), [-1, model_settings.is_weekend_weekday_embedding_size])
            ],
            axis=1
        )
        self.batch_results = tf.concat([self.reduced_day_grained_output_h, self.hour_grained_last_time_outputs, self.prediction_day_inputs], 1)
        self.FCN_input2hidden_params = tf.get_variable(name="FCN_input2hidden_params", shape=[2 * model_settings.day_grained_cell_size
                                                                                              + model_settings.hour_grained_cell_size
                                                                                              + model_settings.day_of_week_embedding_size
                                                                                              + model_settings.holidays_distance_embedding_size
                                                                                              + model_settings.end_of_holidays_distance_embedding_size
                                                                                              + model_settings.is_weekend_weekday_embedding_size,
                                                                                              model_settings.fcn_hidden_layer_size], dtype='float64')
        self.FCN_input_layer_biases = tf.get_variable(name="FCN_input_layer_biases", shape=[model_settings.fcn_hidden_layer_size], dtype='float64')

        self.hidden_layer_input = tf.nn.xw_plus_b(self.batch_results, self.FCN_input2hidden_params, self.FCN_input_layer_biases)
        self.hidden_layer_output = tf.tanh(self.hidden_layer_input)
        self.FCN_hidden2output_params = tf.get_variable(name="FCN_hidden2output_params", shape=[model_settings.fcn_hidden_layer_size, 1], dtype='float64')
        self.FCN_hidden_layer_bias = tf.get_variable(name="FCN_output_layer_bias", shape=[1], dtype='float64')
        self._Y = tf.nn.xw_plus_b(self.hidden_layer_output, self.FCN_hidden2output_params, self.FCN_hidden_layer_bias)
        self.Y = tf.placeholder(name="true_value", shape=[None, 1], dtype='float64')
        self.empirical_loss = tf.losses.mean_squared_error(self.Y, self._Y)
        # tf.get_collection('l2_loss')
        # for v in tf.trainable_variables():
        #     if v is word_embedding:
        #         continue
        #     tf.add_to_collection('l2_loss', v)
        # 这里的l2_lambda应由训练model_settings设置
        # self.l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(model_settings.l2_lambda),
        #                                                       weights_list=tf.get_collection('l2_loss'))
        # 后续会加入结构损失
        self.final_loss = self.empirical_loss
