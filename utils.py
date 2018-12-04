

def generate_feed_dict(model,
                       day_of_week_train_batch,
                       holidays_distance_train_batch,
                       end_of_holidays_distance_train_batch,
                       is_weekend_weekday_train_batch,
                       impression_per_day_train_batch,
                       prediction_day_day_of_week_train_batch,
                       prediction_day_holidays_distance_train_batch,
                       prediction_day_end_of_holidays_distance_train_batch,
                       prediction_day_is_weekend_weekday_train_batch,
                       hour_per_day_train_batch,
                       impression_per_hour_train_batch,
                       Y_train_batch,
                       actual_batch_size,
                       keep_prob):
    feed_dict = dict()
    feed_dict[model.day_of_week] = day_of_week_train_batch
    feed_dict[model.holidays_distance] = holidays_distance_train_batch
    feed_dict[model.end_of_holidays_distance] = end_of_holidays_distance_train_batch
    feed_dict[model.is_weekend_weekday] = is_weekend_weekday_train_batch
    feed_dict[model.impression_per_day] = impression_per_day_train_batch
    feed_dict[model.prediction_day_day_of_week] = prediction_day_day_of_week_train_batch
    feed_dict[model.prediction_day_holidays_distance] = prediction_day_holidays_distance_train_batch
    feed_dict[model.prediction_day_end_of_holidays_distance] = prediction_day_end_of_holidays_distance_train_batch
    feed_dict[model.prediction_day_is_weekend_weekday] = prediction_day_is_weekend_weekday_train_batch
    feed_dict[model.hour_per_day] = hour_per_day_train_batch
    feed_dict[model.impression_per_hour] = impression_per_hour_train_batch
    feed_dict[model.Y] = Y_train_batch
    feed_dict[model.actual_batch_size_scalar] = [actual_batch_size]
    feed_dict[model.keep_prob] = keep_prob
    return feed_dict