def generate_feed_dict(model,
                       day_of_week_batch,
                       holidays_distance_batch,
                       end_of_holidays_distance_batch,
                       is_weekend_weekday_batch,
                       impression_per_day_batch,
                       prediction_day_day_of_week_batch,
                       prediction_day_holidays_distance_batch,
                       prediction_day_end_of_holidays_distance_batch,
                       prediction_day_is_weekend_weekday_batch,
                       hour_per_day_batch,
                       impression_per_hour_batch,
                       Y_batch,
                       actual_batch_size,
                       keep_prob):
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
    feed_dict[model.actual_batch_size_scalar] = [actual_batch_size]
    feed_dict[model.keep_prob] = keep_prob
    return feed_dict