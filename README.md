# Weakly-Supervised-Object-Localization
In this project, we train object detectors with only image-level annotations and no bounding box annotations! First, we used classification models and examine their backbone features for object localization cues.Next, we trained object detectors in the weakly supervised setting, which means we trained an object detector without bounding box annotations!

When we use a classification network like AlexNet, it is trained using a classification loss (cross-entropy). Therefore, in order to minimize this loss function, the network maximizes the likelihood for a given class. Since CNNs preserve spatial locality, this means that the model implicitly learns to produce high activations in the feature map around the regions where an object is present. We used this property to approximately localize the object in the image. This is called a weakly-supervised paradigm: supervised because we have image-level classification labels, but weak since we don't have ground-truth bounding boxes.

We used the PyTorch framework to design our models, train and test them. We also used Weights and Biases for visualizations and to log our metrics.

## Dataset
We trained and tested the model using the [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) data. The Pascal VOC dataset comes with bounding box annotations, however, we did not use bounding box annotations in the weakly-supervised setting.

1. First download the image dataset and annotations. Use the following commands to setup the data, and let's say it is stored at location `$DATA_DIR`.
```bash
$ # First, cd to a location where you want to store ~0.5GB of data.
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ tar xf VOCtrainval_06-Nov-2007.tar
$ # Also download the test data
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar && tar xf VOCtest_06-Nov-2007.tar
$ cd VOCdevkit/VOC2007/
$ export DATA_DIR=$(pwd)
```

## Paper Cited
1. Oquab, Maxime, et al. "Is object localization for free?-weakly-supervised learning with convolutional neural networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
2. Bilen, Hakan, and Andrea Vedaldi. "Weakly supervised deep detection networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016. 

## Results
### Object Localization
Here are a couple examples of object localization using modeified AlexNet backbone. They have been colorized using the 'jet' color

![image](https://user-images.githubusercontent.com/72159394/216883690-1da063c4-7b13-40e7-93ab-256f8541a1d6.png)

### Weakly Supervised Deep Detection Networks 
In WSDDN, the images and region proposals are taken as input into the spatial pyramid pooling (SPP) layer, both classification and detection is done on these regions, and class scores and region scores are computed & combined to predict the best possible bounding box. 

![image](https://user-images.githubusercontent.com/72159394/216884251-3a19e732-2145-4d94-a264-27ec0cc1b285.png)
![image](https://user-images.githubusercontent.com/72159394/216884292-b792cc3a-f1e1-4a9b-9c37-0e350a8d5662.png)

## Limitations of WSDDN
1. Weakly Supervised Deep Detection Networks(WSDDN) rely heavily on region proposal algorithm which makes non-salient object very easily overlooked
2. Region proposal algorithm picks high confident area which often makes the output bounding box smaller than the object
3. Overlook some of objects in the same category 

