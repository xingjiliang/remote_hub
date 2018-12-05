import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('city_id', 1, 'city id')
tf.app.flags.DEFINE_string('train_start_date', '2018-03-19', 'train start date')
tf.app.flags.DEFINE_string('train_end_date', '2018-08-16', 'train end date')
tf.app.flags.DEFINE_bool('load_previous_model', False, 'city id')
tf.app.flags.DEFINE_string('previous_model_name', "", 'previous_model_name')
tf.app.flags.DEFINE_integer('previous_model_epoch_times', 0, 'previous_model_epoch_times')
tf.app.flags.DEFINE_float('learning_rate', 1, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob', 1, 'keep prob')
tf.app.flags.DEFINE_float('l2_lambda', 1e-5, 'l2 lambda')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 100, 'epoch times')
tf.app.flags.DEFINE_bool('evaluate_when_training', True, 'evaluate when training')
tf.app.flags.DEFINE_integer('full_connection_layer_nums', 1, 'full_connection_layer_nums')
tf.app.flags.DEFINE_integer('day_grained_sequence_length', 7, 'day grained sequence length')
tf.app.flags.DEFINE_integer('day_grained_cell_size', 20, 'day_grained_cell_size')
tf.app.flags.DEFINE_integer('day_of_week_embedding_size', 3, 'day_of_week_embedding_size')
tf.app.flags.DEFINE_integer('holidays_distance_size', 7*2+2, 'holidays_distance_size')
tf.app.flags.DEFINE_integer('holidays_distance_embedding_size', 3, 'holidays_distance_embedding_size')
tf.app.flags.DEFINE_integer('end_of_holidays_distance_size', 7+2, 'end_of_holidays_distance_size')
tf.app.flags.DEFINE_integer('end_of_holidays_distance_embedding_size', 3, 'end_of_holidays_distance_embedding_size')
tf.app.flags.DEFINE_integer('is_weekend_weekday_embedding_size', 3, 'is_weekend_weekday_embedding_size')
tf.app.flags.DEFINE_integer('hour_grained_sequence_length', 24, 'hour_grained_sequence_length')
tf.app.flags.DEFINE_integer('hour_grained_cell_size', 10, 'hour_grained_cell_size')
tf.app.flags.DEFINE_integer('hour_per_day_embedding_size', 3, 'hour_per_day_embedding_size')
tf.app.flags.DEFINE_integer('fcn_layer_nums', 1, 'fcn_layer_nums')
tf.app.flags.DEFINE_integer('fcn_hidden_layer_size', 20, 'fcn_hidden_layer_size')

origin_dataset_dir_path = "origin_dataset/"
dataset_dir_path = "dataset/"
model_graph_path = "model_graphs/"
model_path = "models/"
dateset_start_date = '2017-03-03'
mytest_dataset_file_path = origin_dataset_dir_path + "1.tsv"
mytest_train_data_file_path = dataset_dir_path
mytest_test_data_file_path = dataset_dir_path

holidays = [
    '2017-04-02',
    '2017-04-03',
    '2017-04-04',
    '2017-04-29',
    '2017-04-30',
    '2017-05-01',
    '2017-05-28',
    '2017-05-29',
    '2017-05-30',
    '2017-10-01',
    '2017-10-02',
    '2017-10-03',
    '2017-10-04',
    '2017-10-05',
    '2017-10-06',
    '2017-10-07',
    '2017-10-08',
    '2017-12-30',
    '2017-12-31',
    '2018-01-01',
    '2018-02-15',
    '2018-02-16',
    '2018-02-17',
    '2018-02-18',
    '2018-02-19',
    '2018-02-20',
    '2018-02-21',
    '2018-04-05',
    '2018-04-06',
    '2018-04-07',
    '2018-04-29',
    '2018-04-30',
    '2018-05-01',
    '2018-06-16',
    '2018-06-17',
    '2018-06-18',
    '2018-09-22',
    '2018-09-23',
    '2018-09-24',
    '2018-10-01',
    '2018-10-02',
    '2018-10-03',
    '2018-10-04',
    '2018-10-05',
    '2018-10-06',
    '2018-10-07']

end_of_holidays = [
    '2017-04-04',
    '2017-05-01',
    '2017-05-30',
    '2017-10-08',
    '2018-01-01',
    '2018-02-21',
    '2018-04-07',
    '2018-05-01',
    '2018-06-18',
    '2018-09-24',
    '2018-10-07']

weekend_weekdays = [
    '2017-04-01',
    '2017-05-27',
    '2017-09-30',
    '2018-02-11',
    '2018-02-24',
    '2018-04-08',
    '2018-04-28',
    '2018-09-29',
    '2018-09-30']
