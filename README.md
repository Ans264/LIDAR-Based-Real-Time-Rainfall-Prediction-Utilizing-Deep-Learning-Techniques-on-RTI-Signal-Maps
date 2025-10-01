# LIDAR-Based Real-Time Rainfall Prediction Utilizing Deep Learning Techniques on RTI Signal Maps

This repository contains the source code for the paper: "Comparative Assessment of Deep Learning Approaches for Real-Time Rainfall Prediction Using LIDAR-Derived RTI Signal Maps." The project uses deep learning models to predict rainfall conditions from Light Detection and Ranging (LIDAR) data.

-----

## File Descriptions

This repository is organized into data processing scripts (Julia) and modelling scripts (Python).

  * `Preprocessing.jl`: A Julia script for preprocessing the raw Range-Time Intensity (RTI) signal maps from LIDAR data.
  * `RTI_Plotting.jl`: A Julia script for visualizing the processed RTI signal maps.
  * `Baseline CNN.py`: Implements the baseline Convolutional Neural Network (CNN) for the classification task.
  * `VGG16.py`: Implements the VGG16 pre-trained model using a transfer learning approach.
  * `InceptionV3.py`: Implements the InceptionV3 pre-trained model using a transfer learning approach. This model achieved the highest performance.

-----

### Prerequisites

You will need to have Python and Julia installed on your system.

**Python Dependencies:**

  * TensorFlow / Keras
  * NumPy
  * Pandas
  * Matplotlib
  * Scikit-learn

**Julia Dependencies:**

  * Instructions for installing Julia packages can be found in the official [Julia documentation](https://docs.julialang.org/en/v1/stdlib/Pkg/).

-----

## License

This project is licensed under the MIT License.

Copyright (c) 2025 Ansuman Sahu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
