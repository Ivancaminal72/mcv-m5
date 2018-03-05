#!/bin/bash

recorrer_directorio()
{
dir=$(dir)

for file in $dir;
do
    # comprobamos que la cadena no este vacía
    if [ -n $file ]; then
        if [ -d "$file" ]; then
            # si es un directorio, accedemos a él,
            # llamamos recursivamente a la función recorrer_directorio
            #echo "DIR: " $file
            cd $file
            count=$(dir -l|grep -e .png|wc -l)
            echo $file";"$count >> /home/master/code/pre-config/$1.txt
            recorrer_directorio ./
            # una vez que hemos terminado, salimos del directorio (IMPORTANTE)
            cd ..
        #else
            # dividimos la extensión del nombre del fichero y lo mostramos en pantalla
            #extension=${file##*.}
            #path_and_name=${file%.*}
        fi;
    fi;
done;
}

path=/data/module5/Datasets/classification/KITTI/train
cd $path
recorrer_directorio "KITTI_train"

path=/data/module5/Datasets/classification/KITTI/test
cd $path
recorrer_directorio "KITTI_test"

path=/data/module5/Datasets/classification/KITTI/valid
cd $path
recorrer_directorio "KITTI_valid"
