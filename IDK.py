import math
from collections import Counter
import pandas as pd


def Entropy(l):
    return -sum([(i/sum(l))*(math.log2((i/sum(l)))) for i in l])


def attrEntropy(df, attr):
    df1 = df.groupby([attr])['Play cricket'].apply(list)
    netAttrEnt = 0
    for i in range(len(df1)):
        l = df1.iloc[i]
        cnt = Counter(l).values()
        ent = Entropy(cnt)
        netAttrEnt += ((len(l)/len(df))*ent)
    return netAttrEnt


def IG(net, netattr):
    return net - netattr


def best(df):
    net = Entropy(df['Play cricket'].value_counts())
    if (net == 0):
        return list(df['Play cricket'])[0]
    maxi = -1
    for col in df.columns[:-1]:
        val = IG(net, attrEntropy(df, col))
        if (maxi < val):
            maxi = val
            column = col
    return column


def id3(df):
    diction = {}
    highigcol = best(df)
    if highigcol not in list(df.columns):
        return highigcol
    for val in list(set(df[highigcol])):
        copy = df.copy()
        copy = copy[copy[highigcol] == val]
        copy = copy.drop([highigcol], axis=1)
        diction[(highigcol, val)] = id3(copy)
    return diction


def predict(main_dict, input_values):
    key = list(main_dict.keys())[0][0]
    inp = input_values[key]
    branch = (key, inp)
    next_dict = main_dict[branch]
    if type(next_dict) != dict:
        return next_dict
    return predict(main_dict[branch], input_values)


df = pd.read_csv('datasetcsv.csv')
df = df.set_index('Day')
df

output = id3(df)
print(output)

# Prediction for given input
input_values = {'Outlook': 'Sunny', 'Temperature': 'Mild', 'Humidity': 'High'}
prediction = predict(output, input_values)

print('\nPrediction for', input_values, 'is', prediction)
