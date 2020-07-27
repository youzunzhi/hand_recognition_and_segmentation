# README

## Getting Started

### Installation

```
git clone https://github.com/youzunzhi/hand_recognition_and_segmentation.git
```

### Prerequisites

```
python 3.x 
numpy
sklearn
logging
```

### Get Datasets

You can download the hand dataset from [BaiduYun](https://pan.baidu.com/s/1EJBK3O0zYeMafgmktqzlsg ) (Password: jaag) or [Google Drive](https://drive.google.com/drive/folders/1E6gbeIx7hjkKtnSmhRrWpmKNCOMDP78t?usp=sharing).

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

To train SVM model (with RBF Kernel, C=10) using training dataset and evaluate the model on test dataset:

```shell
python svm.py
```

You will get the evaluation result as follows:

```
Acc: 0.8100 (162/200)
Precision: 0.8571 (108/126)
Recall: 0.8438 (108/128)
```



To grid search the hyperparameter choices of SVM, uncomment this [line](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/a615c1b147ede41faf9e020de641ff4f8ea3c958/svm.py#L104) in `svm.py` and run it:

```python
if __name__ == '__main__':
    grid_search_1()
```

The score of each hyperparameter setting will be logged to a `grid_search_1.txt` file in `outputs/svm_grid_search` directory. You can also modify the grid search scheme in `grid_search_1` or `grid_search_2` functions in `svm.py`.



## Segmentation

To segment an image, change the `img_path` to the path of that image in this [line](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/a615c1b147ede41faf9e020de641ff4f8ea3c958/cluster.py#L251) of `cluster.py`, then run this file:

```
python cluster.py
```

which will get the segmentation result of this image in the `outputs/segmentation` directory:

![](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/master/outputs/segmentation/Pic_2018_07_24_100447_blockId%232797_seg_3040.png)

Note that the resolution of the segmentation is so low because we downsampled the image to 30x40 before applying clustering method. It then only needs 0.5-1s to segment. If you change this [line](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/a615c1b147ede41faf9e020de641ff4f8ea3c958/cluster.py#L7) to `60, 80`, it will take 30-40s with probably better result:

![](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/master/outputs/segmentation/Pic_2018_07_24_100447_blockId%232797_seg.png)



To grid search the hyperparameter choices of clustering methods, uncomment this [line](https://github.com/youzunzhi/hand_recognition_and_segmentation/blob/a615c1b147ede41faf9e020de641ff4f8ea3c958/cluster.py#L248) in `cluster.py` and run it:

```python
if __name__ == '__main__':
    grid_search_1()
```

The results will be saved to `outputs/seg_grid_search_1/` directory. You can also modify the `img_path`, `cluster_method`, `init_method_`, `gamma_s` and `gamma_c` to see more segmentation results.