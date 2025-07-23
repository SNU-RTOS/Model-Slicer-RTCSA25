# Model Slicer (from DNNPipe)
Slicing a given DNN model into multiple sub-models based on user-defined layer indices

## Acknowledgement
This code is derived from the original implementation of 
*[DNNPipe: Dynamic Programming-based Optimal DNN Partitioning for Pipelined Inference on IoT Networks]*.

While DNNPipe focuses on optimal automated partitioning using dynamic programming, 
this version provides a manual slicing interface for users to define custom DNN partitioning boundaries for experimentation and prototyping purposes.

DNNPipe: https://www.sciencedirect.com/science/article/pii/S1383762125001341?via%3Dihub

DNNPipe Github Repository: https://github.com/SNU-RTOS/DNNPipe

## System Requirements

- Rubikpi (Debian 13)
- Python 3.10.16
- TensorFlow 2.12.0

## Usage
**The model downloader (`model_download.py`)** : Downloads a pretrained DNN model (ResNet50) for inference
  - Function: Loads the ResNet50 model with pretrained ImageNet weights and saves it in `.h5` format.
  - Output: `resnet50.h5` – Keras H5 format model file (used as input to the slicer)  
  ```bash
  python model_downloader.py 
  ```

**The model partitioner (`model_slicer.py`)** : Interactively slices a given DNN model into multiple sub-models based on user-defined layer indices
  - Input: DNN model in `.h5` format (e.g., `resnet50.h5`)
  - Output: Skiced sub-models in `.tflite` formats
  ```bash
  python model_slicer.py --model-path ./resnet50.h5
  ```

## Example
  ```bash
  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_downloader.py 
  WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.
  Model saved as resnet50.h5

  (.ws_pip) taebikpi@RUBIKPi:~/workspace/DNNPipe_tutorial$ python model_slicer.py --model-path ./resnet50.h5
  How many submodels? 4
  Enter 3 slicing points for ranges: (0, x1), (x1+1, x2), (x2+1, x3), (x3+1, 176)
  Enter x1 x2 x3: 40 80 120
  Slicing ranges: [(0, 40), (41, 80), (81, 120), (121, 176)]
  Saved sliced tflite model 1 to: ./submodels/resnet50/sub_model_1.tflite
  Saved sliced tflite model 2 to: ./submodels/resnet50/sub_model_2.tflite
  Saved sliced tflite model 3 to: ./submodels/resnet50/sub_model_3.tflite
  Saved sliced tflite model 4 to: ./submodels/resnet50/sub_model_4.tflite
  ```
 
