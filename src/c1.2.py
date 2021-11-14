#!/usr/bin/env python3

from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import arff

data = arff.loadarff('data/supermarket.arff')
df = pd.DataFrame(data[0])

# print(df.head())
# print(df.shape)
# print(df.dtypes)

# convert categorical values into one-hot vectors and ignore ? values
# corresponding to missing values
supermarket_one_hot = pd.get_dummies(df)
supermarket_one_hot.drop(supermarket_one_hot.filter(regex='_b\'\?\'$',axis=1).columns,axis=1,inplace=True)

itemsets = apriori(supermarket_one_hot, min_support=0.1)

# option to show all itemsets
pd.set_option('display.max_colwidth',None)

itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
# print(itemsets[['support', 'length']])

rules = association_rules(itemsets, min_threshold=0.7, metric='confidence')

# print(rules.head(50))

# for index,row in rules.iterrows():
#     print(row)

filtered_rules = rules.loc[map(lambda x: len(x)==4, rules['antecedents'])]
filtered_rules = filtered_rules.loc[map(lambda x: len(x)==1, filtered_rules['consequents'])]
print(filtered_rules)

