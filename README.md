# Complex-Valued Neural Networks in Keras with Tensorflow
[![Documentation](https://readthedocs.org/projects/keras-complex/badge/?version=latest)](https://keras-complex.readthedocs.io/) [![PyPI Status](https://img.shields.io/pypi/status/keras-complex.svg)](https://pypi.python.org/pypi/keras-complex) [![PyPI Versions](https://img.shields.io/pypi/pyversions/keras-complex.svg)](https://pypi.python.org/pypi/keras-complex) [![Build Status](https://travis-ci.org/JesperDramsch/keras-complex.svg?branch=main)](https://travis-ci.org/JesperDramsch/keras-complex) [![PyPI License](https://img.shields.io/pypi/l/keras-complex.svg)](LICENSCE.md)


[Complex-valued convolutions](https://en.wikipedia.org/wiki/Convolution#Domain_of_definition) could provide some interesting results in signal processing-based deep learning. A simple(-ish) idea is including explicit phase information of time series in neural networks. This code enables complex-valued convolution in convolutional neural networks in [keras](https://keras.io) with the [TensorFlow](https://tensorflow.org/) backend. This makes the network modular and interoperable with standard keras layers and operations.

This code is very much in **Alpha**. Please consider helping out improving the code to advance together. This repository is based on the code which reproduces experiments presented in the paper [Deep Complex Networks](https://arxiv.org/abs/1705.09792). It is a port to Keras with Tensorflow-backend.

Requirements
------------

- numpy
- scipy
- scikit-learn
- keras
- tensorflow 2.X or tensorflow-gpu 2.X

Install requirements for computer vision experiments with pip:
```
pip install -f requirements.txt
```

For the non-gpu version:
```
pip install -f requirements-nogpu.txt
```

Depending on your Python installation you might want to use anaconda or other tools.


Installation
------------

```
pip install keras-complex
```
and
```
pip install tensorflow-gpu
```

Usage
-----
Build your neural networks with the help of keras. 

```python
import complexnn

import keras
from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()

model.add(complexnn.conv.ComplexConv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 2)))
model.add(complexnn.bn.ComplexBatchNormalization())
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.compile(optimizer=optimizers.Adam(), loss='mse')

```

An example working implementation of an autoencoder can be found [here](https://github.com/JesperDramsch/Complex-CNN-Seismic/).

Citation
--------

Find the [CITATION file](/CITATION.cff) or cite this software version as:
```
@misc{dramsch2019complex, 
    title     = {Complex-Valued Neural Networks in Keras with Tensorflow}, 
    url       = {https://figshare.com/articles/Complex-Valued_Neural_Networks_in_Keras_with_Tensorflow/9783773/1}, 
    DOI       = {10.6084/m9.figshare.9783773}, 
    publisher = {figshare}, 
    author    = {Dramsch, Jesper S{\"o}ren and Contributors}, 
    year      = {2019}
}
```

Please cite the original work as: 

```
@ARTICLE {Trabelsi2017,
    author  = "Chiheb Trabelsi, Olexa Bilaniuk, Ying Zhang, Dmitriy Serdyuk, Sandeep Subramanian, João Felipe Santos, Soroush Mehri, Negar Rostamzadeh, Yoshua Bengio, Christopher J Pal",
    title   = "Deep Complex Networks",
    journal = "arXiv preprint arXiv:1705.09792",
    year    = "2017"
}
```
