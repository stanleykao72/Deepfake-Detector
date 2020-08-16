from bot.credentials import bot_token, bot_user_name, telegram_user_id, URL
import os

import aiohttp
from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage, SendPhoto
from aiogram.utils.executor import start_webhook
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
import aiogram.utils.markdown as md
from aiogram.types import ParseMode
import time
import json
from datetime import datetime

import imageio
#from predict.tl_inception_resnet import load_model, inference
#from detect_from_video import test_full_image_network, load_model

import asyncio

import logging


global TOKEN
TOKEN = bot_token
# PROJECT_NAME = os.getenv('PROJECT_NAME', 'aiogram-example')  # Set it as you've set TOKEN env var

WEBHOOK_HOST = URL  # Enter here your link from Heroku project settings
WEBHOOK_PATH = '/webhook/'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = int(os.environ.get('PORT', 5000))

# websocket
WS_DOMAIN = os.getenv("WS_DOMAIN", "localhost")
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", 8888))
# global channel_id
# channel_id = ''

LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.info(f'PORT:{WEBAPP_PORT}')

bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
media = types.MediaGroup()


# load model
data_path = './predict/data/'

output_path = './output/'
#load_model(model_path, cuda=False)

# dp = Dispatcher(bot)
# dp.middleware.setup(LoggingMiddleware())


# States
class VideoForm(StatesGroup):
    model = State()
    Video = State()

ws_post = {}


async def fetch_wsurl(session, url, headers=None):
    async with session.get(url, headers=headers) as response:
        json = await response.json()
        wsurl = json["url"]
        channel_id = json["channel_id"]
        return wsurl, channel_id


async def get_channel():
    session = aiohttp.ClientSession()
    ws_url, channel_id = await fetch_wsurl(
        session,
        f"http://{WS_DOMAIN}:{WS_PORT}/api/rtm.connect",
        headers={"Authorization": "Token Wcw33RIhvnbxTKxTxM2rKJ7YJrwyUXhXn"},
    )
    logger.info(ws_url)
    return session, ws_url, channel_id


async def websocket_conn(session, ws_url):
    async with session.ws_connect(ws_url) as ws:
        logger.info(f"Client connected to {ws_url}")

        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                if msg.data == 'close cmd':
                    await ws.close()
                    break
                else:
                    await ws.send_str(msg.data)
            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break
            logger.info(f"Message received: {msg.data}")
            if msg.data != 'Done!!':
                try:
                    logger.info(f'msg.data!=Done###{msg.data}###')
                    params = json.loads(msg.data)
                    message_id = params["message_id"]
                    predit_video = params["predit_video"]
                    chat_id = params["chat_id"]
                    logger.info(f'message_id:{message_id}, predit_video:{predit_video}')
                except Exception as e:
                    logger.info(e)

            else:
                t = time.time()
                logger.info(f"detected during {t - ws_post[types.User.get_current().id]['start_time']:.2f} seconds.")
                try:
                    await bot.send_message(chat_id,
                                           f"detected during {t - ws_post[types.User.get_current().id]['start_time']:.2f} seconds.",
                                           reply_to_message_id=message_id)
                    await bot.send_document(chat_id,
                                            types.InputFile(predit_video),
                                            reply_to_message_id=message_id)
                    await ws.close()
                    ws_post.pop(types.User.get_current().id)

                except Exception as e:
                    logger.info(e)


@dp.message_handler(commands='websocket')
@dp.async_task
async def on_websocket(message: types.Message):
    logger.info('start....wait for websocket')
    logger.info(f'chat id in on_websocket: {message.chat.id}')
    session, ws_url, channel_id = await get_channel()

    logger.info(f'session:{session}')
    logger.info(f'ws_url:{ws_url}')
    logger.info(f'channel_id:{channel_id}')

    post_url = f"http://{WS_DOMAIN}:{WS_PORT}/api/rtm.push/{channel_id}"

    try:
        ws_post[types.User.get_current().id] = {"chat_id": types.Chat.get_current(), "session": session, "ws_url": ws_url, "post_url": post_url, "channel_id": channel_id}
    except Exception as e:
        logger.error(e)

    logger.info(f'User:{types.User.get_current()}')
    logger.info(f'Chat:{types.Chat.get_current()}')
    #logger.info(f'ws_post in on_websocket: {ws_post[types.User.get_current().id]}')
    logger.info('connecting to  websocket server')
    await websocket_conn(session, ws_url)
    logger.info('session is closing')
    await session.close()

# @dp.message_handler(commands='start')
# async def cmd_start(message: types.Message):
#     print('start....wait for websocket')
#     session, ws_url, channel_id = await get_channel()
#     await websocket_conn(session, ws_url)
#     print('connected to  websocket server')


@dp.message_handler(commands='video')
async def cmd_video(message: types.Message):
    username = message.from_user.full_name
    logger.info(f'chat id in cmd_video: {message.chat.id}')
    try:
        logger.info(f'ws_post in cmd_video: {ws_post[types.User.get_current().id]}')
        logger.info(f'chat_id in cmd_video: {message.chat.id}:{ws_post[types.User.get_current().id]["chat_id"].id}')
        logger.info(f'session in cmd_video: {message.chat.id}:{ws_post[types.User.get_current().id]["session"]}')
    except Exception as e:
        logger.error(e)
    if ws_post[types.User.get_current().id]["session"]:
        # Set state
        await VideoForm.model.set()
        await message.reply(f"{username}, this is DeepFake detector.... ")
    else:
        await message.reply(f"{username}, please send /websocket command first, thanks!")

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add("EfficientnetB7", "SPPNet", "XceptionNet")

    await message.reply("Please choose a model from BUTTON", reply_markup=markup)


@dp.message_handler(lambda message: message.text not in ["EfficientnetB7", "SPPNet", "XceptionNet"], state=VideoForm.model)
async def process_model_invalid(message: types.Message, state: FSMContext):
    """
    In this example gender has to be one of: EfficientnetB7, SPPNet, XceptionNet.
    """
    logger.info(f'not in EfficientnetB7/SPPNET/XceptionNext......')
    logger.info(f'chat id in process_model_invalid: {message.chat.id}')

    return await message.reply("Bad controller. Choose EfficientnetB7/SPPNET/XceptionNet from the BUTTON.")


@dp.message_handler(lambda message: message.text in ["EfficientnetB7", "SPPNet", "XceptionNet"], state=VideoForm.model)
async def process_model(message: types.Message, state: FSMContext):
    logger.info(f'chat id in process_model: {message.chat.id}')

    async with state.proxy() as datavideo:
        datavideo['model'] = message.text
        logger.info(f'model: {message.text}......')
    # Remove keyboard
    markup = types.ReplyKeyboardRemove()
    logger.info(markup)

    logger.info(f'state:{await state.get_state()}')
    # And send message
    await bot.send_message(message.chat.id,
                           md.text(md.text('Please upload your video(mp4) for detecting (limited 20M), thanks!'),
                                   sep='\n',
                                   ),
                           reply_markup=markup,
                           parse_mode=ParseMode.MARKDOWN,
                           )
    await VideoForm.next()
    logger.info(f'state next:{await state.get_state()}')


# @dp.message_handler(content_types=types.ContentTypes.ANY, state=VideoForm.Video)
# @dp.async_task
# async def process_video_invalid(message: types.Message, state: FSMContext):
#     """
#     Process upload Video
#     """
#     logger.info(f'chat id in process_video: {message.chat.id}')
#     logger.info(message)
#     try:
#         if message.video:
#             logger.info(message.video.mime_type)
#         if message.animation:
#             logger.info(message.animation.mime_type)
#     except Exception as e:
#         logger.error(e)


@dp.message_handler(content_types=types.ContentTypes.ANY, state=VideoForm.Video)
@dp.async_task
async def process_video(message: types.Message, state: FSMContext):
    """
    Process upload Video
    """
    logger.info(f'chat id in process_video: {message.chat.id}')
    logger.info(f'conent_types: {types.ContentTypes}')
    try:
        logger.info(f'ws_post in process_video: {ws_post[types.User.get_current().id]}')
        logger.info(f'chat_id in process_video: {message.chat.id}:{ws_post[types.User.get_current().id]["chat_id"].id}')
        logger.info(f'session in process_video: {message.chat.id}:{ws_post[types.User.get_current().id]["session"]}')
        logger.info(f'ws_url in process_video: {message.chat.id}:{ws_post[types.User.get_current().id]["ws_url"]}')
    except Exception as e:
        logger.info(e)

    async with state.proxy() as datavideo:
        logger.info(datavideo['model'])
        model_name = datavideo['model']
        if model_name == 'SPPNet':
            model_path = './pretrained_model/SPP-res50.pth'
        if model_name == 'XceptionNet':
            model_path = './pretrained_model/df_c0_best.pkl'
        if model_name == 'EfficientnetB7':
            model_path = './pretrained_model/b7_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_0'

    logger.info(f'10......0')

    logger.info(message)
    try:
        if message.video:
            logger.info('message.video')
            logger.info(message.video.mime_type)
            logger.info(message.video)
            logger.info(message.video.file_id)
            file_id = message.video.file_id
        elif message.animation:
            logger.info('message.animation')
            logger.info(message.animation.mime_type)
            logger.info(message.animation)
            logger.info(message.animation.file_id)
            file_id = message.animation.file_id
        # message.document:
        else:
            logger.info("it's not a mp4 video, please upload a mp4 video")
    except Exception as e:
        logger.error(e)

    filename = await bot.get_file(file_id)
    file_path = filename.file_path

    logger.info(f'20......')
    videoname = data_path + file_id + '.mp4'
    await bot.download_file_by_id(file_id, destination=videoname, seek=True)

    #vid = imageio.get_reader(videoname)
    #print(f'vid:{vid}')

    logger.info(f'25......')
    async with state.proxy() as datavideo:
        datavideo['Video'] = message.animation

    logger.info(f'30......')
    # await message.reply(filename)
    # await message.reply(file_path)
    # await message.reply_photo(file_id)

    # ##### inference #########
    # await message.reply(f'This image is {inference(model, data_path, imagename)}')
    video_path = videoname
    predit_video = output_path + file_id + '.mp4'
    await message.reply("wait for detecting video......................")
    param_json = {"model_name" : model_name
                  , "video_path" : video_path
                  , "model_path" : model_path
                  , "output_path" : output_path
                  , "threshold" : 0.5
                  , "start_frame" : 0
                  , "end_frame" : None
                  , "cuda" : False
                  , "chat_id" : message.chat.id
                  , "message_id" : message.message_id
                  , "predit_video" : predit_video}

    logger.info(f'35......')
    try:
        ws_post[types.User.get_current().id]["start_time"] = time.time()
    except Exception as e:
        logger.info(e)
    try:
        ws_session = ws_post[types.User.get_current().id]["session"]
        ws_url = ws_post[types.User.get_current().id]["ws_url"]
        channel_id = ws_post[types.User.get_current().id]["channel_id"]
        post_url = ws_post[types.User.get_current().id]["post_url"]
        logger.info(f'wsurl:{ws_url}')
        async with ws_session.post(post_url, json=param_json) as resp:
            logger.info(await resp.text())
    except Exception as e:
        logger.info(f'ws_session:{e}')
    # await test_full_image_network(video_path, model_path, output_path,
    #                               start_frame=0, end_frame=None, cuda=False)

    logger.info(f'40......')
    logger.info(f'predit video:{predit_video}')

    # Finish conversation
    await state.finish()


async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    # insert code here to run it after start


async def on_shutdown(dp):
    logging.warning('Shutting down..')

    # insert code here to run it before shutdown

    # Remove webhook (not acceptable in some cases)
    await bot.delete_webhook()

    # Close DB connection (if used)
    await dp.storage.close()
    await dp.storage.wait_closed()

    logging.warning('Bye!')


if __name__ == '__main__':
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        skip_updates=True,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT
    )
