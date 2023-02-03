# Weakly-Supervised-Object-Localization
In this project, we train object detectors with only image-level annotations and no bounding box annotations! First, we used classification models and examine their backbone features for object localization cues.Next, we trained object detectors in the weakly supervised setting, which means we trained an object detector without bounding box annotations!

When we use a classification network like AlexNet, it is trained using a classification loss (cross-entropy). Therefore, in order to minimize this loss function, the network maximizes the likelihood for a given class. Since CNNs preserve spatial locality, this means that the model implicitly learns to produce high activations in the feature map around the regions where an object is present. We used this property to approximately localize the object in the image. This is called a weakly-supervised paradigm: supervised because we have image-level classification labels, but weak since we don't have ground-truth bounding boxes.

We used the PyTorch framework to design our models, train and test them. We also used Weights and Biases for visualizations and to log our metrics.
