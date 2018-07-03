
# Goal
The objective of this project is to have a working example of a TensorRT model written with the C++ TensorRT API, running inference in half precision mode.
To the best of our knowledge, there is no public example of API-based model inferencing in half precision; while in most cases we managed to run half precision inference when parsing models from other frameworks such as caffe or tensorflow, we fail with model natively written with the API.

# Content
* `demo.cpp` - model definition and inference
* `wts_gen_demo.py` - weight file conversion from general dictionary of numpy array to TensorRT wts format, either in full or half precision
* `./images` - test images to run the inference
* `./data` - data folder containing weights both in pickle dictionary format and in TensorRT wts format
* `Makefile`

# Dependencies
* OpenCV >= 2.4
* TensorRT RC 4.0.0.3
* CUDA 9.0

# Requirements
A NVIDIA GPU with FP16 support is needed






