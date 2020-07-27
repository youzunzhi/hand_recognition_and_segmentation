# README

## Getting Started

### Installation

```
git clone https://github.com/youzunzhi/hand_recognition_and_segmentation.git
```

### Prerequisites

```
python 3.x 
sklearn
PyTorch >= 1.0.1 
yacs
logging
```

### Get Datasets

You can download the hand dataset from [BaiduYun](https://pan.baidu.com/s/1EJBK3O0zYeMafgmktqzlsg ) (Password: jaag) or Google Drive.

(If you are using Linux Terminal, you can follow the tutorial from [here](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99) to download the files shared on Google Drive with `wget`.)

After downloading the dataset, please unzip all the zip files and move them to the `dataset` directory. The `dataset` directory's hierarchy should be as follow:

```
dataset/
	1_without_hand/
		Pic_2018_07_25_093931_blockId#28929.bmp
		...
	2_with_hand/
		Pic_2018_07_25_094725_blockId#40120.bmp
		...
	...
	test_data/
		with_hand/
		without_hand/
	training_dataset.txt
	testing_dataset.txt
```

## Recognition


