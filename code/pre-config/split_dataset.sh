 #!/bin/bash


function moveFiles() folder folder2 {

	cd $folder

	num_files = ls | wc -l 
	filenames = dir $folder

	if [num_files<1] then
		numTrainSet = num_files/70
		numTestSet = num_files - NumTrainSet
		for i in filenames
		do 
			if [$i>$numTrainSet] then 
				mv  ${numeros[i]}  $files2
		done

		

	else
		echo "Empty folder..."
	fi

	cd ..
}


# create test folder and subfolders
mkdir test
mkdir pedestrian
mkdir background
cd ..

# acced train folder
cd train 
moveFiles(pedestrian, test/pedestrian)
moveFiles (background, test/background)
