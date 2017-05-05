#!/bin/bash

echo 'removing old data directory ...'
rm -rf ./data 2>/dev/null
mkdir data

preprocessing=../preprocessing
initEmbedding=$preprocessing/init-embedding/initSenseEmbedding.txt
networkData=$preprocessing/idmapping/triple2id.txt
entityData=$preprocessing/idmapping/entity2id.txt
relationData=$preprocessing/idmapping/relation2id.txt

echo 'copying ${initEmbedding} ...'
cp $initEmbedding data/
echo 'copying ${networkData} ...'
cp $networkData data/
echo 'copying ${entityData} ...'
cp $entityData data/
echo 'copying ${relationData} ...'
cp $relationData data/
