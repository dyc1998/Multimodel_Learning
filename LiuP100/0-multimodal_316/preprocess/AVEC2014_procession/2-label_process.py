import os
import pandas
import csv
def label(root,label_txt):
    for csv in os.listdir(root):
        print("csv:",csv)
        data_csv=os.path.join(root,csv)
        df=pandas.read_csv(data_csv,header=None)
        print(df[0][0])
        with open(label_txt,'a') as af:
            af.write(csv[:5]+';'+str(df[0][0])+'\n')

root="/usr/data/local/duanyuchi/python_data/AVEC_2014/source_data/label/DepressionLabels/"
label_txt='/usr/data/local/duanyuchi/python_data/AVEC_2014/label.txt'
label(root,label_txt)