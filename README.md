# AEDNet: Asynchronous Event Denoising with Spatial-Temporal Correlation among Irregular Data
Source code for [AEDNet paper](https://dl.acm.org/doi/10.1145/3503161.3548048) from ACMMM 2022

A novel asynchronous event denoising neural network directly consumes the correlation of the irregular signal in spatiotemporal range without destroying its original structural property.

<img src="https://github.com/Fanghuachen/AEDNet/blob/AEDNet/pic/gif%20.gif" width="400" height="300"> <img src="https://github.com/Fanghuachen/AEDNet/blob/main/pic/gif1.gif" width="400" height="300"> 

If you find this work useful in your academic reaserch, please cite the following work:
```
@inproceedings{fang2022aednet,
  title={AEDNet: Asynchronous Event Denoising with Spatial-Temporal Correlation among Irregular Data},
  author={Fang, Huachen and Wu, Jinjian and Li, Leida and Hou, Junhui and Dong, Weisheng and Shi, Guangming},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={1427--1435},
  year={2022}
}
```

If there is any suggestion or questions, feel free to fire an issue to let us know. :)

# Installation

This code was tested on an Ubuntu 20.04.1 system (i9-10920X CPU, 128GB RAM, and GeForce RTX 3090Ti GPU) running Python 3.7, Pytorch 1.12 and CUDAToolkit 11.3.1.
```
conda create --name aednet python=3.7
conda activate aednet
pip install numpy=1.21.6
pip install matplotlib=3.5.2
pip install tensorboardX
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

# Train the model
We have released the trained model parameters in Releases. If you want to train your own model, you should first put your data in "data/AEDNetDataset_BA" and list the name of data in "training_file_name.txt". After that, you can train your model via:
```
python train_net.py --trainset training_file_name.txt --testset test_file_name.txt --nepoch 2000 --batchSize 8
```

# Test the model
Put the model parameters in Releases to "models/BA_noise_removal_model" and test it via:
```
python test_net.py --shapename MAH00444_50{i} --x_frame 1280 --y_frame 720
```

# DVSCLEAN Dataset
To download the dataset use:[DVSCLEAN](https://drive.google.com/file/d/14FJD-kf9NA-bdWVWHK35ewLiLVdBnNSq/view?usp=share_link)



