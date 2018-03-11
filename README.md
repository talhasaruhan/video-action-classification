# Video Action Classification Using Spatial Temporal Clues

Original paper: arXiv:1504.01561

This is a Tensorflow implementation of the paper "Modeling Spatial-Temporal Clues in a Hybrid Deep
Learning Framework for Video Classification"

* **optical_flow.py** : computes optical flow from data and resizes videos into 224x224 for compatibility with pre-trained VGG network.
* **model.py** : Tensorflow implementation of the paper
* **cnnm.py** : Cnn-M architecture implementation
* **vgg.py** : Vgg19 architecture implementation

As a modification to the original paper, instead of creating prediction scores with Spatial CNN, Motion CNN and Regularized Fusion Network seperately and combining them, I concatenated them into one feature vector and trained using only one softmax layer. This is a mathematical generalization of the originally proposed method, so with enough data and time, it should perform at least as good as the original model. 

If you want the original model, youy can simply create three softmax layers and linearly combine their prediction scores (weights selected either through backprop or hand selected using cross-validation) for the final prediction.

Cnn-M and Vgg19 implementations support memory efficient weight loading using placeholders to avoid copying constant numpy arrays  throught the graph. (Takes 2X memory rather than 3X when copying data from the disk to the C backend). Both models also support saving weights into a numpy dumps.

Vgg19 implementation is based on https://github.com/machrisaa/tensorflow-vgg. But in order to allow loading/saving w/ placeholders, the model has been heavily modified thus pre-trained weights have a different internal structure to the base model above.
