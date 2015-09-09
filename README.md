# DeepLearning-shugyou

My personal training to understand Deep Learning deeply...

## Setup

1. Install [Anaconda](https://store.continuum.io/cshop/anaconda/), a comfy Python environment for data processing.
   You can find an installer on the above site.
2. Install [PyBrain](http://pybrain.org/), a Neural Network library.

    ```bash
    ~/anaconda/bin/pip install git+https://github.com/pybrain/pybrain.git
    ```
    
    Note: I encountered errors when using pybrain package in PyPI.
    I recommend to install package in GitHub.
3. Install Benchmarker. Some codes use it.

    ```bash
    ~/anaconda/bin/pip install Benchmarker
    ```

## Shugyou01 - PyBrain Tutorial

Write (or copy-and-paste) **working codes** from [PyBrain tutorial](http://pybrain.org/docs/index.html#tutorials).

## Shutyou02 - Logic Element

Learns logic element's function such as XOR.

- xor.py: Provides basic examples of DNN.
- xor_best_nn.py: Trying a few kinds of DNNs to find the fastest and the most precise one.
