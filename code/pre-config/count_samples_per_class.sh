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
            count=$(dir | wc -l)
            echo $file";"$count >> /home/master/ivan/pre-config/$1.txt
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

path=/data/module5/Datasets/classification/TT100K_trafficSigns/train
cd $path
recorrer_directorio "train"

path=/data/module5/Datasets/classification/TT100K_trafficSigns/test
cd $path
recorrer_directorio "test"

path=/data/module5/Datasets/classification/TT100K_trafficSigns/valid
cd $path
recorrer_directorio "valid"
