import csv
import pandas as pd
import numpy as np

data_path = '2dietadj.csv'
data_label = '2dietadj'

if data_path[-1] == 't':  # txt
    dataset = pd.read_csv(data_path, sep='\t')
elif data_path[-1] == 'v':  # csv
    dataset = pd.read_csv(data_path, sep=',')
elif data_path[-1] == 'z':  # .txt.gz
    dataset = pd.read_csv(data_path, sep='\t', compression='gzip')
else:
    raise Exception('Unrecognized File Type')

data_headers = dataset.columns.values
dataset = dataset.values
c_index = np.where(data_headers == 'Cluster')[0][0]
c_values = dataset[:, c_index]
unique_c_values = np.array(list(set(list(c_values))))
data_headers = np.delete(data_headers,c_index)
dataset = np.delete(dataset,c_index,axis = 1)

for u in unique_c_values:
    with open(data_label+'_'+str(u[1:])+'.csv', mode="w") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(data_headers)
        counter = 0
        for row in c_values:
            if row == u:
                writer.writerow(dataset[counter])
            counter += 1
    file.close()


