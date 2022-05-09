# NLP assignment 2
# Task A

import pandas as pd
import csv

# 1: Class distributions 
with open('data/olid-train.csv') as csv_file:
    test_data = pd.read_csv(csv_file)

# print occurences of labels 
print(test_data['labels'].value_counts())

# compute relative frequencies 
tot_len = len(test_data)
rel_occ0 = test_data['labels'].value_counts()[0]/tot_len
rel_occ1 = test_data['labels'].value_counts()[1]/tot_len
print('freq 0:', rel_occ0, 'and freq 1:', rel_occ1)

