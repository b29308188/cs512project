#!/bin/bash

if [ "$#" -ne 1 ]; then
	echo "usage: ./runEmbedding.sh [dataSetName]" 
	exit
fi

inputDataDir=./data/$1
intermediateDir=./tmp/$1-intermediate
outputDir=./results/$1

rm -r $intermediateDir 2>/dev/null
rm -r $outputDir 2>/dev/null

mkdir $intermediateDir
mkdir $outputDir

python3 ./bin/initEmbedding.py $inputDataDir $intermediateDir -d 300 

./bin/Train  $intermediateDir $outputDir
