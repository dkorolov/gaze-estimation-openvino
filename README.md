# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
Project was tested in MacOS. Same set up instruction can be used for Linux, and with some small changes it can works on Windows

####1. OpenVINO instalation
Download OpenVINO instalation files from [here](https://docs.openvinotoolkit.org/latest/index.html) and follow original [instructions](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_macos.html)

####2. Virtual environment set up
Recuirements: Python3 should be installed. I set my Virtualenv in the folder ```~/env/openvino_gaze``` in code below. Path to virtual environment can be changed if need.

```
cd ~/env
virtualenv -p python3 ./openvino_gaze
cd openvino_gaze
source bin/activate

#install libraries requirements from project:
pip install -r <path to project>/requirements.txt
```
####3. Download the models
Download pre-trained models from [Intel Model Zoo](https://docs.openvinotoolkit.org/latest/omz_models_intel_index.html) using the [OpenVINO model downloader](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html). We need to download next models:

* [Face Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Head Pose Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Gaze Estimation](https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

All models was downloaded to folder ```<path to project>/gaze-estimation-openvino/models ``` Models downloading code:

```
cd ~/env/openvino_gaze/
source bin/activate

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --help

cd <path to project>/gaze-estimation-openvino

# face-detection-adas-0001 model downloading (FP32, FP16, INT8)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 --precisions FP32 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 --precisions FP16 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-0001 --precisions INT8 -o ./models

# face-detection-adas-binary-0001 model downloading (FP32)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001  -o ./models

# landmarks-regression-retail-0009 model downloading (FP32, FP16)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP32 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009 --precisions FP16 -o ./models

# head-pose-estimation-adas-0001 model downloading (FP32, FP16)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP32 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001 --precisions FP16 -o ./models

# gaze-estimation-adas-0002 model downloading (FP32, FP16)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP16 -o ./models

```

## Demo
*TODO:* Explain how to run a basic demo of your model.

```
# gaze-estimation-adas-0002 model downloading (FP32, FP16)
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP32 -o ./models
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002 --precisions FP16 -o ./models

```

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
