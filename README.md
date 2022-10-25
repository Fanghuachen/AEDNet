# AEDNet: Asynchronous Event Denoising with Spatial-Temporal Correlation among Irregular Data
Source code for [AEDNet paper](https://dl.acm.org/doi/10.1145/3503161.3548048) from ACMMM 2022

A novel asynchronous event denoising neural network directly consumes the correlation of the irregular signal in spatiotemporal range without destroying its original structural property.

<img src="https://github.com/Fanghuachen/AEDNet/blob/main/pic/gif%20.gif" width="400" height="300"> <img src="https://github.com/Fanghuachen/AEDNet/blob/main/pic/gif1.gif" width="400" height="300"> 

**Installation**

This code was tested on an Ubuntu 20.04.1 system (i9-10920X CPU, 128GB RAM, and GeForce RTX 3090Ti GPU) running Python 3.7, Pytorch 1.12 and CUDAToolkit 11.3.1.
```
conda create --name aednet python=3.7
conda activate aednet
pip install numpy==1.21.6
pip install matplotlib=3.5.2
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
