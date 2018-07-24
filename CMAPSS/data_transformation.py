import csv
import numpy as np

with open('train_FD001.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line for line in stripped if line)
    grouped = [line.split(',') for line in lines]
    with open('train_FD001.csv', 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(grouped)

filename = 'train_FD001.csv'
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=' ', quoting=csv.QUOTE_NONE)
x = list(reader)
data = np.array(x).astype('float')
print(data)