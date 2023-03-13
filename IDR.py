import math
from collections import Counter
import pandas as pd


def calculate_entropy(counter):
    total = sum(counter)
    return round(sum([-val/total * math.log(val/total, 2) for val in counter]), 3)


def calculate_information_gain(main_df, column, label_column):
    label_counter = main_df[label_column].value_counts()
    net_entropy = calculate_entropy(label_counter)
    categories = list(set(main_df[column]))
    grouped = main_df.groupby([column])[label_column].apply(list)
    information_gain = net_entropy
    for category, values in grouped.iteritems():
        counter = Counter(values).values()
        entropy = calculate_entropy(counter)
        print(f'\tCategory: {category}, Entropy: {entropy}')
        information_gain -= entropy * sum(counter) / len(main_df)
    return information_gain


def best_column_ID3(main_df, label_column):
    best_column = ''
    best_info_gain = -1.0
    for column in main_df.columns[:-1]:
        print('\nCalculating for', column)
        information_gain = calculate_information_gain(
            main_df, column, label_column)
        print('\nInformation Gain:', information_gain)
        if information_gain == 0:
            return list(main_df[label_column])[0]
        if best_info_gain < information_gain:
            best_info_gain = information_gain
            best_column = column

    return best_column


def ID3(main_df, label_column):
    best_column = best_column_ID3(main_df, label_column)
    columns = main_df.columns
    if best_column not in columns:
        return best_column
    categories = list(set(main_df[best_column]))
    main_dict = {}
    for category in categories:
        copy_df = main_df.copy()
        copy_df = copy_df[copy_df[best_column] == category]
        copy_df = copy_df.drop([best_column], axis=1)
        main_dict[(best_column, category)] = ID3(copy_df, label_column)
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
final_dict = ID3(main_df, label_column)
print(final_dict)

# Prediction for given input
input_values = {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High'}
prediction = predict(final_dict, input_values)

print('\nPrediction for', input_values, 'is', prediction)
