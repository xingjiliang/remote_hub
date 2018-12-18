import numpy as np


def generate_feed_dict(data_batch, model, keep_prob):
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
    goal_batch = []
    for record in data_batch:
        day_of_week_batch.append(record[1][:, 0])
        holidays_distance_batch.append(record[1][:, 1])
        end_of_holidays_distance_batch.append(record[1][:, 2])
        is_weekend_weekday_batch.append(record[1][:, 3])
        impression_per_day_batch.append(record[1][:, 5])
        prediction_day_day_of_week_batch.append(record[2][0])
        prediction_day_holidays_distance_batch.append(record[2][1])
        prediction_day_end_of_holidays_distance_batch.append(record[2][2])
        prediction_day_is_weekend_weekday_batch.append(record[2][3])
        hour_per_day_batch.append(record[3][:, 0])
        impression_per_hour_batch.append(record[3][:, 2])
        Y_batch.append(record[4])
        goal_batch.append(record[5])
    day_of_week_batch = np.array(day_of_week_batch, dtype='int32')
    holidays_distance_batch = np.array(holidays_distance_batch, dtype='int32')
    end_of_holidays_distance_batch = np.array(end_of_holidays_distance_batch, dtype='int32')
    is_weekend_weekday_batch = np.array(is_weekend_weekday_batch, dtype='int32')
    impression_per_day_batch = np.array(impression_per_day_batch, dtype='float64').reshape([-1, 7, 1])
    prediction_day_day_of_week_batch = np.array(prediction_day_day_of_week_batch, dtype='int32').reshape([-1, 1])
    prediction_day_holidays_distance_batch = np.array(prediction_day_holidays_distance_batch, dtype='int32').reshape(
        [-1, 1])
    prediction_day_end_of_holidays_distance_batch = np.array(prediction_day_end_of_holidays_distance_batch,
                                                             dtype='int32').reshape([-1, 1])
    prediction_day_is_weekend_weekday_batch = np.array(prediction_day_is_weekend_weekday_batch, dtype='int32').reshape(
        [-1, 1])
    hour_per_day_batch = np.array(hour_per_day_batch, dtype='int32')
    impression_per_hour_batch = np.array(impression_per_hour_batch, dtype='float64').reshape([-1, 24, 1])
    Y_batch = np.array(Y_batch, dtype='float64').reshape([-1, 1])
    goal_batch = np.array(goal_batch, dtype='float64').reshape([-1, 4])
    feed_dict = dict()
    feed_dict[model.day_of_week] = day_of_week_batch
    feed_dict[model.holidays_distance] = holidays_distance_batch
    feed_dict[model.end_of_holidays_distance] = end_of_holidays_distance_batch
    feed_dict[model.is_weekend_weekday] = is_weekend_weekday_batch
    feed_dict[model.impression_per_day] = impression_per_day_batch
    feed_dict[model.prediction_day_day_of_week] = prediction_day_day_of_week_batch
    feed_dict[model.prediction_day_holidays_distance] = prediction_day_holidays_distance_batch
    feed_dict[model.prediction_day_end_of_holidays_distance] = prediction_day_end_of_holidays_distance_batch
    feed_dict[model.prediction_day_is_weekend_weekday] = prediction_day_is_weekend_weekday_batch
    feed_dict[model.hour_per_day] = hour_per_day_batch
    feed_dict[model.impression_per_hour] = impression_per_hour_batch
    feed_dict[model.Y] = Y_batch
    feed_dict[model.self.actual_batch_size] = Y_batch.shape[0]
    feed_dict[model.keep_prob] = keep_prob
    return feed_dict, goal_batch
