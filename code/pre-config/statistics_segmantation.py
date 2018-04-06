import os
import sys
import imp
import csv
import numpy as np
from PIL import Image

config_path = '/data/module5/Datasets/segmentation/'+sys.argv[1]+'/config.py'
path = '/data/module5/Datasets/segmentation/'+sys.argv[1]+'/'+sys.argv[2]+'/masks/'
dc=imp.load_source('config', config_path)
print(sys.argv[1]+'/'+sys.argv[2])
print('-------------------------------')
classCount = {}
classSize = {}
files = os.listdir(path)
for file in files:
    if '.png' in file:
        im_frame = Image.open(path+file)
        np_frame = np.array(im_frame.getdata())
        current_count = np.bincount(np_frame)
        for i in dc.classes.keys():
            if len(current_count)<i+1:
                c = 0
            else:
                c = current_count[i]

            if dc.classes[i] in classCount.keys():
                classCount[dc.classes[i]] += c
                classSize[dc.classes[i]].append(c)
            else:
                classCount[dc.classes[i]] = c
                classSize[dc.classes[i]] = [c]


w = csv.writer(open(sys.argv[1]+"_"+sys.argv[2]+".csv","w"))
# Headers



#min_val = np.min(np.min())
#max_val = np.max(np.max([classSize[key] for key in classCount.keys()]))
#med_val = np.median(np.median([classSize[key] for key in classCount.keys()]))


#obj_size = [min_val+step/2.0+float(n)*step for n in range(N)]
#w.writerow(["Class","Number pix","count per image"])
w.writerow(["Class" ,"Number pix","pix_per_img", "num of imgs"])
w.writerow(["","",dc.img_shape[0]*dc.img_shape[1], len(files)])


for key,val in classCount.items():
    val_current = np.array(classSize[key])
    row = [key,val]
    #row.append(val_current.tolist())
    w.writerow(row)

print(dc.img_shape)

w2 = csv.writer(open(sys.argv[1]+"_"+sys.argv[2]+"_hist.csv","w"))
# Headers



#min_val = np.min(np.min())
#max_val = np.max(np.max([classSize[key] for key in classCount.keys()]))
#med_val = np.median(np.median([classSize[key] for key in classCount.keys()]))


#obj_size = [min_val+step/2.0+float(n)*step for n in range(N)]
#w.writerow(["Class","Number pix","count per image"])
w2.writerow(classCount.keys())

for i in range(len(classSize[classCount.keys()[0]])):
    row = []
    for key,val in classSize.items():
        row.append(val[i])
    w2.writerow(row)

print(dc.img_shape)
