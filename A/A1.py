import os
import os.path
import numpy as np
import pandas as pd
import pickle
# root_dir = 'C:\\Users\\SU DEHONG\\Desktop\\test'
# file_name = 'test.txt'
# path = os.path.join(root_dir,file_name)
#
# file = open(path,'rb')
# data1 = pickle.load(file)

#number and smile
path1 = 'C:\\Users\\SU DEHONG\\Desktop\\test\\B\\cas20210902.txt'
with open(path1) as f:
    l1 = f.readlines()
dic = {}
for i in range(len(l1)):
    l2 = l1[i].split()
    l = l2[1].split('.')
    # print(l2[0],l[0])
    dic.update({l2[0]:l[0]})
print(dic.keys())

#number and
pickle.load
rs=pickle.load(open("test1.txt", "rb"))
# print(rs)

key = rs.keys()
val = rs.values()

tableA = pd.DataFrame({'mol_id':dic.keys(),'smile':dic.values()})
print(tableA)
tableB = pd.DataFrame({'mol_id':rs.keys(),'energy':rs.values()})
print(tableB)

table = tableA.merge(tableB,on='mol_id',how='left')
print(table)

table.to_csv(r'data.csv')