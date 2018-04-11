import os
import sys
import imp

config_path = '/data/module5/Datasets/detection/'+sys.argv[1]+'/config.py'
path = '/data/module5/Datasets/detection/'+sys.argv[1]+'/'+sys.argv[2]+'/'
dc=imp.load_source('config', config_path)

classCount = {}
files = os.listdir(path)
for file in files:
    if '.txt' in file:
        with open(path+file) as doc:
            for line in doc:
                if dc.classes[int(line.split()[0])] in classCount.keys():
                    classCount[dc.classes[int(line.split()[0])]] += 1
                else:
                    classCount[dc.classes[int(line.split()[0])]] = 1

print(classCount)
