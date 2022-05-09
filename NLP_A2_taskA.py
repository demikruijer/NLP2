# NLP assignment 2
# Task A

from random import sample
import pandas as pd
import csv

# 1: Class distributions 
with open('data1/olid-train.csv') as csv_file:
    test_data = pd.read_csv(csv_file)

# print occurences of labels 
print(test_data['labels'].value_counts())

# compute relative frequencies 
tot_len = len(test_data)
rel_occ0 = test_data['labels'].value_counts()[0]/tot_len
rel_occ1 = test_data['labels'].value_counts()[1]/tot_len
print('freq not:', rel_occ0, 'and freq off:', rel_occ1)

# print random tweet for both classes
sample_off = test_data[test_data.labels == 1].sample(n=1)
print('offensive example:', sample_off.iloc[0]['text'])
sample_not = test_data[test_data.labels == 0].sample(n=1)
print('non-offensive example:', sample_not.iloc[0]['text'])