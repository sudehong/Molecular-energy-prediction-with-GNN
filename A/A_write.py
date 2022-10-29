import os
import os.path
import numpy as np
import pickle


def string_to_float(str):
    return float(str)

def find_small_energy(path,Maker):
    #read the file
    with open(path) as f:
        l1 = f.readlines()
    #find the smaller eneger and force in each line
    energy = []
    min = float(1)
    e = float(0)
    force = []
    is_found = 0
    l6 = []
    for i in range(len(l1)):

        if (l1[i].strip() == 'Item               Value     Threshold  Converged?'):
            l2 = l1[i + 1].split()
            l3 = l1[i + 2].split()
            l4 = l1[i + 3].split()
            l5 = l1[i + 4].split()
            if l2[4] == 'YES' and l3[4] == 'YES' and l4[4] == 'YES' and l5[4] == 'YES':
                for k in range(i,0,-1):
                    l6 = l1[k].split()
                    if not l6:
                        continue
                    if l6[0] == 'SCF' :
                        e = string_to_float(l6[4])
                        energy.append(e)
                        is_found = 1
                        break
    if is_found == 1:
        return energy
    elif is_found == 0:
        return 0
#
Maker = '%nprocshared=6'
# root_dir = 'E:\done\\00'
# file_name = '10124203_r00'
# file_name = '12065_r01'
# path = os.path.join(root_dir,file_name + '.out')
#
# energy_1 = find_small_energy(path,Maker)
# print(energy_1)

file_dir = '/public/yasudak/sakai/data/dft/done'
energy_small = {}
for root, dirs, files in os.walk(file_dir):
    # print(root)
    # print(files)
    for file in files:
        if file.endswith('.out'):
            path = os.path.join(root, file)
            energy_1 = find_small_energy(path, Maker)
            if energy_1 == 0 :
                small = 'no found energy'
            else :
                small = min(energy_1)

            ss = str(file).split('_')
            dic = {ss[0]:str(small)}
            energy_small.update(dic)


f =  open('test1.txt','wb')
pickle.dump(energy_small,f)
f.close()

# pickle.load
# rs=pickle.load(open("test.txt", "rb"))
# print(rs)






