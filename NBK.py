from collections import Counter
import pandas as pd


def counterDict(df):
    counter = {}
    output = dict(df['Play cricket'].value_counts(dropna=False))
    counter.update(output)
    for col in df.columns[:-1]:
        df1 = df.groupby([col])['Play cricket'].apply(list)
        for val in range(len(df1)):
            dic = Counter(df1.iloc[val])
            counter.update({list(df1.index)[val]: dict(dic)})
    return counter


def calcProb(l, output, cntDict, df):
    prob = {}
    for out in output:
        p = 1
        for val in l:
            p *= (cntDict[val][out]/cntDict[out])
        p = p*(cntDict[out]/len(df))
        prob.update({out: p})
    return prob


def naiveBayes(df, ques):
    countDict = counterDict(df)
    Prob = calcProb(ques, list(set(df['Play cricket'])), countDict, df)
    total = sum(Prob.values())
    maxKey = ''
    maxVal = 0
    for key, val in Prob.items():
        if val / total > maxVal:
            maxKey = key
            maxVal = val / total
    return maxKey, round(maxVal, 4)


df = pd.read_csv('datasetcsv.csv')
df = df.set_index('Day')
k, v = naiveBayes(df, ['Sunny', 'Mild', 'High', 'Weak'])
print("For ['Sunny','Mild','High','Weak'], the output will be ", k, " with probability ", v)
