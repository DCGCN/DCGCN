# DSTGNN_
This is a PyTorch implementation of the paper: Dynamic Spatio-Temporal Graph Neural Network for Traffic Forecasting. 

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Preparation

### METR-LA and PEMS-BAY
Download the METR-LA and PEMS-BAY dataset from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) provided by [Li et al.](https://github.com/liyaguang/DCRNN.git) . Move them into the data folder. 


## Model Training and Evaluate

* METR-LA

```
python train.py
```
* PEMS-BAY

```
python train.py --traffic_file ./data/PEMS-BAY/pems-bay.h5 --SE_file ./data/PEMS-BAY/SE\(PEMSBAY\).txt --model_file data/PEMS-BAY/PEMS-BAY --log_file data/PEMS-BAY/logbay
```

### Run the Pre-trained Model
set the epoch=100 in yaml.
* METR-LA

```
python test.py
```
* PEMS-BAY

```
python test.py --traffic_file ./data/PEMS-BAY/pems-bay.h5 --SE_file ./data/PEMS-BAY/SE\(PEMSBAY\).txt --model_file data/PEMS-BAY/PEMS-BAY --log_file data/PEMS-BAY/logbay
```

