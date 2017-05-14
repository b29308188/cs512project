#!/bin/bash

rawData=./wikiData
senseFile=$rawData/sense.init.300d.txt
results=./wiki-results

rm -r $results 2>/dev/null

mkdir $results

embeddingExe=./bin/initEmbedding.py
filterExe=./bin/filterProcessing.py


echo 'first filter entiteis that does not exist in glove and print transformed id mappings to files'
python $filterExe  $rawData $senseFile $results
echo 'based on new entity mapping create init embedding file'
python $embeddingExe $results/entity2id.txt $senseFile $results
