import os
import sys
import imp
import csv
import numpy as np
config_path = '/data/module5/Datasets/detection/'+sys.argv[1]+'/config.py'
path = '/data/module5/Datasets/detection/'+sys.argv[1]+'/'+sys.argv[2]+'/'
dc=imp.load_source('config', config_path)

classCount = {}
classSize = {}
files = os.listdir(path)
for file in files:
    if '.txt' in file:
        with open(path+file) as doc:
            for line in doc:
                if dc.classes[int(line.split()[0])] in classCount.keys():
                    classCount[dc.classes[int(line.split()[0])]] += 1
                    classSize[dc.classes[int(line.split()[0])]].append(float("{0:.2f}".format(1000.0*float(line.split()[3])*float(line.split()[4]))))
                else:
                    classCount[dc.classes[int(line.split()[0])]] = 1
                    classSize[dc.classes[int(line.split()[0])]] = [float("{0:.2f}".format(1000.0*float(line.split()[3])*float(line.split()[4])))]


w = csv.writer(open(sys.argv[1]+"_"+sys.argv[2]+".csv","w"))
# Headers
#all_val = [classSize[key] for key in classCount.keys()]
all_val = []
for v in [classSize[key] for key in classCount.keys()]:
    all_val.extend(v)
all_val.sort()
min_val = all_val[0]
max_val = all_val[len(all_val)-int(0.15*len(all_val))]
max_val_real = all_val[len(all_val)-1]


#min_val = np.min(np.min())
#max_val = np.max(np.max([classSize[key] for key in classCount.keys()]))
#med_val = np.median(np.median([classSize[key] for key in classCount.keys()]))

N = 40
step = (max_val-min_val)/float(N)
step = 0.8
N = int((max_val_real-min_val)/step)
#obj_size = [min_val+step/2.0+float(n)*step for n in range(N)]
w.writerow(["Class","Number of Objects","Object-Size(mili-%)","Count"])

for key,val in classCount.items():
    val_current = np.array(classSize[key])
    val_current.sort()
    H = []*N
    obj_size = [0]
    for i in range(N):
        if i==N:
            IDX = np.where(val_current is not None)[0]
            obj_size.append(max_val_real)
        else:
            IDX = np.where(val_current <= ((float(i)+1.0)*step)**2)[0]
            obj_size.append(((float(i)+1.0)*step)**2)
        H.append(np.shape(IDX)[0])
        val_current[IDX] = None

    w.writerow([key,val,obj_size,H])

print(classCount)
#print(classSize)
