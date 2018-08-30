# AAD: Adaptive Anomaly Detection System

## Introduction

This project is a anomaly detction system which is using optical flow and object detction (FASTER RCNN). Please read [Report.pdf](https://github.com/mfarhadi/Anomaly_Detection/blob/master/Report.pdf) file.

For a quick overview on the AAD results please watch [my video](https://youtu.be/SlCy7cPocQU) 

## Preparation 


First of all, clone the code
```
git clone https://github.com/mfarhadi/Anomaly_Detection.git
```

Then, create a folder:
```
mkdir data
```

### prerequisites

* Python 2.7
* Pytorch 0.2.0
* CUDA 8.0 or higher


### Pretrained Model

* ResNet101: [GDrive](https://drive.google.com/file/d/1TE_1S5K4RMyJ-eh5qlvlhN65C20OhYrp/view?usp=sharing)

Download it and put it into the models/ folder.




### Compilation

As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |
  
More details about setting the architecture can be found [here](https://developer.nvidia.com/cuda-gpus) or [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

Install all the python dependencies using pip:
```
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```
cd lib
sh make.sh
```

It will compile all the modules you need, including NMS, ROI_Pooing, ROI_Align and ROI_Crop. The default version is compiled with Python 2.7, please compile by yourself if you are using a different python version.

**As pointed out in this [issue](https://github.com/jwyang/faster-rcnn.pytorch/issues/16), if you encounter some error during the compilation, you might miss to export the CUDA paths to your environment.**


## Demo

If you want to run detection on your own videos with a pre-trained model, download the pretrained model listed above and put your video inside video folder.

python demo.py 

Then you will find the detection results.



