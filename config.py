import sys
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('remains', True, 'predict remains of one day or full day')
tf.app.flags.DEFINE_string('model_name', 'rnnV2', 'model name')
tf.app.flags.DEFINE_integer('city_id', 1, 'city id')
tf.app.flags.DEFINE_string('train_start_date', '2018-03-19', 'train start date')
tf.app.flags.DEFINE_string('train_end_date', '2018-08-16', 'train end date')
tf.app.flags.DEFINE_bool('load_previous_model', False, 'load_previous_model')
tf.app.flags.DEFINE_integer('previous_model_epoch_times', None, 'previous_model_epoch_times')
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
tf.app.flags.DEFINE_float('l2_lambda', 1e-5, 'l2 lambda')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch size')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'epoch times')
tf.app.flags.DEFINE_bool('evaluate_when_training', True, 'evaluate when training')
tf.app.flags.DEFINE_integer('test_data_batch_size', 64, 'batch size')
tf.app.flags.DEFINE_string('test_start_date', '2018-08-16', 'test start date')
tf.app.flags.DEFINE_string('test_end_date', '2018-09-21', 'test end date')

origin_dataset_dir = "origin_dataset"
dataset_dir = "dataset"
model_graph_dir = "model_graphs"
model_params_dir = "models"
dateset_start_date = '2017-03-03'

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
