# Deepfake-Detector
------------------
## Create new environment in Anaconda
`conda create -n Deepfake-Detector python=3.7`

## Install & Requirements
The code has been tested on pytorch=1.6.0 and python 3.7, please refer to `requirements.txt` for more details.
### To install the python packages
`python -m pip install -r requiremnets.txt`

## Pretrained Model

1. [XceptionNet]: (https://drive.google.com/drive/folders/1GNtk3hLq6sUGZCGx8fFttvyNYH8nrQS8?usp=sharing) based on FaceForensics++
2. [Resnet+SPPNet] : (https://drive.google.com/file/d/1PyKhf8QX13wQuE4wXNCBlaHypKWjNzq5/view?usp=sharing)
3. [EfficientnetB7+SPPNet] : (https://drive.google.com/file/d/1x_FNPs6x73bUwmNiR7vAUDhFrLtsrlwV/view?usp=sharing)

download and put them in .\pretrained_model folder

## Usage
**To test with videos**

`python detect_from_video.py --video_path ./videos/003_000.mp4 --model_path ./pretrained_model/df_c0_best.pkl -o ./output --cuda`

**To test with images**

`python test_CNN.py -bz 32 --test_list ./data_list/Deepfakes_c0_299.txt --model_path ./pretrained_model/df_c0_best.pkl`

**To train a model**

`python train_CNN.py`
(Please set the arguments after read the code)

## About
If our project is helpful to you, we hope you can star and fork it. If there are any questions and suggestions, please feel free to contact us.

Thanks for your support.
## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.
