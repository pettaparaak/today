import pandas as pd
import os
import math
from collections import Counter


def calculate_entropy(counter):
    total = sum(counter)
    return round(sum([-val/total * math.log(val/total, 2) for val in counter]), 3)


def hunts_algorithm(main_df, label_column):
    counter = Counter(main_df[label_column]).values()
    entropy = calculate_entropy(counter)
    if entropy == 0:
        return list(main_df[label_column])[0]
    columns = main_df.columns[:-1]
    current_column = columns[0]
    categories = list(set(main_df[current_column]))
    main_dict = {}
    for category in categories:
        copy_df = main_df[main_df[current_column] == category].copy()
        copy_df = copy_df.drop([current_column], axis=1)
        main_dict[(current_column, category)] = hunts_algorithm(
            copy_df, label_column)
    return main_dict


def predict(main_dict, input_values):
    key = list(main_dict.keys())[0][0]
    inp = input_values[key]
    branch = (key, inp)
    next_dict = main_dict[branch]
    if type(next_dict) != dict:
        return next_dict
    return predict(main_dict[branch], input_values)


main_df = pd.read_csv('datasetcsv.csv')
main_df = main_df.set_index('Day')
label_column = 'Play cricket'
final_dict = hunts_algorithm(main_df, label_column)
print(final_dict)

# Prediction for given input
input_values = {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High'}
prediction = predict(final_dict, input_values)

print('\nPrediction for', input_values, 'is', prediction)
