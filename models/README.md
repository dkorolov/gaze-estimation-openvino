
## Download the models
Download pre-trained models from Intel model zoo using the OpenVINO model downloader. We need to download next models:

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
