import datetime
import geopy.distance
import numpy as np
import os
import pandas as pd

from sklearn.preprocessing import OneHotEncoder


def load_raw_data(path, type):
    files = os.listdir(os.path.join(path, type))
    data = {}
    surfix = f"_{type}.csv"
    for file in files:
        name = file[:file.find(surfix)]
        df = pd.read_csv(os.path.join(path, type, file), sep=";")
        data[name] = df
    return data

def get_nearest_cities(city_attributes):
    nearest_cities = {}
    for _, row in city_attributes.iterrows():
        source = (row["Latitude"], row["Longitude"])
        city_dist = []
        for _, row2 in city_attributes.iterrows():
            if row["City"] is row2["City"]:
                continue
            destination = (row2["Latitude"], row2["Longitude"])
            city_dist.append((row2["City"], geopy.distance.geodesic(source, destination).km))
        city_dist.sort(key=lambda x: x[1])
        nearest_cities[row["City"]] = [cd[0] for cd in city_dist[:3]]
    return nearest_cities

def load_train_data():
    dict = load_raw_data("data", "train")
    dict.pop("weather_description")
    for key, df in dict.items():
        dict[key] = df.iloc[12:, :]
    return dict

def load_test_data():
    dict = load_raw_data("data", "test")
    dict.pop("weather_description")
    for key, df in dict.items():
        dict[key] = df.iloc[:-1, :]
    return dict

def get_normalization_params(raw):
    params = {}
    for key, df in raw.items():
        all = np.reshape(df.to_numpy()[:, 1:], -1)
        params[key] = (np.nanmin(all), np.nanmax(all))
    return params

def to_city_time_vector(raw):
    cities = next(iter(raw.values())).columns[1:]
    hours = next(iter(raw.values()))[["datetime"]]
    ctvs = {c: hours.copy() for c in cities}
    for city in cities:
        for key, df in raw.items():
            ctvs[city][key] = df[[city]]
    return ctvs

def normalize(ctv, params):
    for df in ctv.values():
        for param, ms in params.items():
            min, max = ms
            df[param] = (df[param] - min) / (max - min)

def denormalized(arr, params):
    min, max = params
    arr = arr * (max - min) + min
    return arr

def aggregate(ctv):
    for key, value in ctv.items():
        df = pd.DataFrame()
        for column in value.columns:
            if column == "datetime":
                df[column] = value[column].groupby(value.index // 3).nth(1)
            else:
                df[column] = value[column].groupby(value.index // 3).mean()
        ctv[key] = df

def normalize_city_attributes(city_attributes):
    latitude_min = city_attributes["Latitude"].min()
    latitude_max = city_attributes["Latitude"].max()

    longitude_min = city_attributes["Longitude"].min()
    longitude_max = city_attributes["Longitude"].max()

    city_attributes["Latitude"] = (city_attributes["Latitude"] - latitude_min) / (latitude_max - latitude_min)
    city_attributes["Longitude"] = (city_attributes["Longitude"] - longitude_min) / (longitude_max - longitude_min)

def to_city_day_vector(ctv, wind_treshold, step=24):
    cdv = {}
    for city, df in ctv.items():
        u = 0
        cdr = []
        while u < len(df):
            w = u + step
            date = df.iloc[u, 0]
            vec = np.reshape(df.iloc[u : w, 1:].to_numpy(), -1)
            temp_mean = df.iloc[u : w]["temperature"].mean()
            wind_cat = int(np.any(df.iloc[u:w]["wind_speed"].to_numpy() > wind_treshold))
            cdr.append((date, vec, temp_mean, wind_cat))
            u = w
        cdv[city] = cdr
    return cdv

def to_city_day_vector2(ctv, wind_treshold, step=24):
    cdv = {}
    for city, df in ctv.items():
        u = 0
        cdr = []
        while u < len(df):
            w = u + step
            date = df.iloc[u, 0]
            vec = np.reshape(df.iloc[u : w, 1:].to_numpy(), -1)
            temp = df.iloc[u : w]["temperature"]
            wind_cat = [int(x) for x in df.iloc[u:w]["wind_speed"].to_numpy() > wind_treshold]
            cdr.append((date, vec, temp, wind_cat))
            u = w
        cdv[city] = cdr
    return cdv

def get_city_encoder(cities_attr):
    cities = np.reshape(cities_attr["City"].to_numpy(), (-1, 1))
    cohe = OneHotEncoder()
    cohe.fit(cities)
    return cohe

def get_wind_treshold(souce, params):
    mean, std = params["wind_speed"]
    return (souce - mean) / std

def drop_nan_records(data_set):
    mask = [np.any(np.isnan(val), axis=0) for val in data_set.values()]
    mask = np.vstack(mask)
    mask = np.any(mask, axis=0)
    return {key: val[..., ~mask] for key, val in data_set.items()}

def drop_nan_records2(data_set):
    mask = [np.any(np.isnan(val), axis=0) for val in data_set.values()]
    mask[4] = np.any(mask[4], axis=0)
    mask = np.vstack(mask)
    mask = np.any(mask, axis=0)
    return {key: val[..., ~mask] for key, val in data_set.items()}

def get_set1(cdv, city_encoder, city_attributes_raw):
    d1 = []
    d2 = []
    d3 = []
    output_temp = []
    output_wind = []
    date = []
    city_one_hot = []
    coords = []

    for city, dv in cdv.items():
        d1 += [r[1] for r in dv[:-4]]
        d2 += [r[1] for r in dv[1:-3]]
        d3 += [r[1] for r in dv[2:-2]]
        output_temp += [r[2] for r in dv[4:]]
        output_wind += [np.hstack((r[3], 1 - r[3])) for r in dv[4:]]
        date_str = [r[0] for r in dv[4:]]
        date += [datetime.datetime.strptime(d, "%d.%m.%Y %H:%M").timetuple().tm_yday / 365 for d in date_str]
        size = len(date_str)
        city_one_hot += [city_encoder.transform([[city]]).toarray()[0]] * size
        coords += [city_attributes_raw.loc[city_attributes_raw["City"] == city][["Latitude", "Longitude"]].to_numpy()] * size

    set = {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "output_temp": output_temp,
        "output_wind": output_wind,
        "date": date,
        "city_one_hot": city_one_hot,
        "coords": coords
    }

    return {key: np.vstack(val).T for key, val in set.items()}

def get_set2(cdv, city_encoder, city_attributes_raw):
    d1 = []
    d2 = []
    d3 = []
    output_temp = []
    output_wind = []
    date = []
    city_one_hot = []
    coords = []

    for city, dv in cdv.items():
        d1 += [r[1] for r in dv[:-4]]
        d2 += [r[1] for r in dv[1:-3]]
        d3 += [r[1] for r in dv[2:-2]]
        output_temp += [r[2].to_numpy() for r in dv[4:]]
        output_wind += [np.dstack((np.asarray(r[3]), 1 - np.asarray(r[3]))) for r in dv[4:]]
        date_str = [r[0] for r in dv[4:]]
        date += [datetime.datetime.strptime(d, "%d.%m.%Y %H:%M").timetuple().tm_yday / 365 for d in date_str]
        size = len(date_str)
        city_one_hot += [city_encoder.transform([[city]]).toarray()[0]] * size
        coords += [city_attributes_raw.loc[city_attributes_raw["City"] == city][["Latitude", "Longitude"]].to_numpy()] * size

    set = {
        "d1": d1,
        "d2": d2,
        "d3": d3,
        "output_temp": output_temp,
        "output_wind": output_wind,
        "date": date,
        "city_one_hot": city_one_hot,
        "coords": coords
    }

    return {key: np.vstack(val).T for key, val in set.items()}

def get_sets__without_neighbors__one_prediction__without_aggregation():
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set, normalization_params)

def get_sets__without_neighbors__one_prediction__with_aggregation():
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    aggregate(train_ctv)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold, 8)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    aggregate(test_ctv)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold, 8)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set, normalization_params)

def get_sets__without_neighbors__24_predictions__without_aggregation():
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector2(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set2(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records2(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector2(test_ctv, wind_treshold)
    test_set = get_set2(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records2(test_set)

    return (train_set, test_set, normalization_params)

def get_sets__without_neighbors__8_predictions__with_aggregation():
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    aggregate(train_ctv)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector2(train_ctv, wind_treshold, 8)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set2(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records2(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    aggregate(test_ctv)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector2(test_ctv, wind_treshold, 8)
    test_set = get_set2(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records2(test_set)

    return (train_set, test_set, normalization_params)

def get_sets__with_3_neighbors__one_prediction__without_aggregation():  # TODO
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    nearest_cities = get_nearest_cities(city_attributes_raw)
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set)

def get_sets__with_3_neighbors__one_prediction__with_aggregation():  # TODO
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    nearest_cities = get_nearest_cities(city_attributes_raw)
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set)

def get_sets__with_3_neighbors__24_predictions__without_aggregation():  # TODO
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    nearest_cities = get_nearest_cities(city_attributes_raw)
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set)

def get_sets__with_3_neighbors__8_predictions__with_aggregation():  # TODO
    city_attributes_raw = pd.read_csv("data/city_attributes.csv", sep=";")

    train_raw = load_train_data()
    nearest_cities = get_nearest_cities(city_attributes_raw)
    normalization_params = get_normalization_params(train_raw)
    train_ctv = to_city_time_vector(train_raw)
    normalize(train_ctv, normalization_params)
    normalize_city_attributes(city_attributes_raw)
    wind_treshold = get_wind_treshold(6, normalization_params)
    train_cdv = to_city_day_vector(train_ctv, wind_treshold)
    city_encoder = get_city_encoder(city_attributes_raw)
    train_set = get_set1(train_cdv, city_encoder, city_attributes_raw)
    train_set = drop_nan_records(train_set)

    test_raw = load_test_data()
    test_ctv = to_city_time_vector(test_raw)
    normalize(test_ctv, normalization_params)
    test_cdv = to_city_day_vector(test_ctv, wind_treshold)
    test_set = get_set1(test_cdv, city_encoder, city_attributes_raw)
    test_set = drop_nan_records(test_set)

    return (train_set, test_set)
