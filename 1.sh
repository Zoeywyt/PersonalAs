#!/bin/bash
# clone internlm-base conda env to user's conda env
# created by xj on 01.07.2024
# modifed by xj on 01.19.2024 to fix bug of conda env clone 

echo "start cloning conda environment of internlm-base"

if [ -z "$1" ]; then
    echo "Error: No argument provided. Please provide a conda environment name."
    exit 1
fi


# echo "uncompress pkgs.tar.gz file to /root/.conda/pkgs..."
# sleep 3
# tar --skip-old-files -xzvf /root/share/pkgs.tar.gz -C /root/.conda

echo "----------"
echo "start create the new conda env: $1"
sleep 3
conda create -n $1 --clone C:/Users/zoey/Desktop/大模型开发/QAnythingDemo/conda_envs/internlm-base
echo "Finised!"
echo "Now you can use your environment by typing:"
echo "conda activate $1"