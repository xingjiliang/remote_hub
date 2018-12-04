import argparse
import sys


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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", default="SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT",
                        type=str, help="Path of train data")
    parser.add_argument("--test_path", default="SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT",
                        type=str, help="Path of test data")
    parser.add_argument("--max_sentence_length", default=90,
                        type=int, help="Max sentence length in data")
    parser.add_argument("--dev_sample_percentage", default=0.1,
                        type=float, help="Percentage of the training data to use for validation")
    parser.add_argument("--hidden_size", default=100,
                        type=int, help="Dimensionality of RNN hidden (default: 100)")
    parser.add_argument("--rnn_dropout_keep_prob", default=0.7,
                        type=float, help="Dropout keep probability of RNN (default: 0.7)")
    parser.add_argument("--desc", default="",
                        type=str, help="Description for model")
    parser.add_argument("--dropout_keep_prob", default=0.5,
                        type=float, help="Dropout keep probability of output layer (default: 0.5)")
    parser.add_argument("--l2_reg_lambda", default=1e-5,
                        type=float, help="L2 regularization lambda (default: 1e-5)")
    parser.add_argument("--batch_size", default=10,
                        type=int, help="Batch Size (default: 10)")
    parser.add_argument("--num_epochs", default=100,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=10,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=100,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=5,
                        type=int, help="Number of checkpoints to store (default: 5)")
    parser.add_argument("--learning_rate", default=1.0,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=0.9,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")
    parser.add_argument("--checkpoint_dir", default="",
                        type=str, help="Checkpoint directory from training run")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=True,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=False,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=True,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    return args