# Video Action Classification Using Spatial Temporal Clues

Original paper: arXiv:1504.01561

This is a Tensorflow implementation of the paper "Modeling Spatial-Temporal Clues in a Hybrid Deep
Learning Framework for Video Classification"

* optical_flow.py : computes optical flow from data and resizes videos into 224x224 for compatibility with pre-trained VGG network.
* model.py : Tensorflow implementation of the paper
* cnnm.py : Cnn-M architecture implementation
* vgg.py : Vgg19 architecture implementation

Cnn-M and Vgg19 implementations support memory efficient weight loading using placeholders to avoid copying constant numpy arrays  throught the graph. Both models also support saving weights into a numpy dumps.

Vgg19 implementation is based on https://github.com/machrisaa/tensorflow-vgg. But in order to allow memory efficient loading/saving, the model has been heavily modified thus pre-trained weights have a different internal structure to the base model above.
