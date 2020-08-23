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

## Telegram BOT
[Telegram BOT reference](https://core.telegram.org/bots#6-botfather)
1. Search BotFather and add him
2. send message /newbot, to create a BOT 
> BotFather: Alright, a new bot. How are we going to call it? Please choose a name for your bot.
>
> xxxBot
>
> Good. Now let's choose a username for your bot. It must end in `bot`. Like this, for example: TetrisBot or tetris_bot.

> xxx_bot    ==> for bot_user_name
3. You will got a TOKEN like '110201543:XXHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw'  ==> for bot_token
4. cp ./bot/credentials_template.py to ./bot/credentials.py, and edit credentials.py

> import os
> 
> bot_token = '110201543:XXHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw'
>
> bot_user_name = 'xxx_bot'
>
> URL = "https://cf45064e05ed.ngrok.io"

5. You can `ngrok` to get URL, and replace URL in the credentials.py

## Start service
Use two terminals and send below command:
1. python app.py
2. python wsserver.py

## Usage
**To test with videos**
in the telegram bot(xxxBot):
1. send /websocket message to startup websocket connection
2. send /video message to choose model and put video in the BOT
3. BOT will reply two videos for Real/Fake video and Grad-cam video

## About
If our project is helpful to you, we hope you can star and fork it. If there are any questions and suggestions, please feel free to contact us.

Thanks for your support.
## License
The provided implementation is strictly for academic purposes only. Should you be interested in using our technology for any commercial use, please feel free to contact us.
