import asyncio
import os
import secrets
import weakref

import aiohttp.web
from aiohttp import web
import aioredis
from aiohttp import WSCloseCode
import json
from detector_inference import detector_inference
import logging

LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
DATE_FORMAT = '%Y%m%d %H:%M:%S'
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(__name__)

WS_DOMAIN = os.getenv("WS_DOMAIN", "localhost")
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", 9999))

routes = web.RouteTableDef()

# app.router.add_get("/ws/{channel_id}", ws_handler)
@routes.get('/ws/{channel_id}')
async def ws_handler(request):
    channel_id = request.match_info.get("channel_id")
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    # 我們直接將連線物件 reference 存放在程式記憶體區塊中
    request.app["websockets"][channel_id] = ws

    logger.info(f"Client connected: {channel_id}")

    try:
        await asyncio.gather(
            listen_to_websocket(ws, request.app, channel_id), listen_to_redis(request.app, channel_id), return_exceptions=True
        )
    finally:
        request.app["websockets"].pop(channel_id, None)
        logger.info("Websocket connection closed")


# app.router.add_get("/api/rtm.connect", wsticket_handler)
@routes.get('/api/rtm.connect')
async def wsticket_handler(request):
    # 實作可在此檢查認證資訊, 通過後產生一個時效性 ticket, 儲存至認證用快取/資料庫等, 供其他服務查閱
    # 最後回覆 websocket 的時效性 url
    channel_id = secrets.token_urlsafe(32)
    logger.info(f"/api/rtm.connect: channel_id -> {channel_id}")
    ws_url = (
        f"ws://{WS_DOMAIN}:{WS_PORT}/ws/{channel_id}"
    )  # 生產環境應為加密的 wss
    logger.info(f"/api/rtm.connect: ws_url -> {ws_url}")

    return aiohttp.web.json_response({"url": ws_url, "channel_id": channel_id})

# app.router.add_post("/api/rtm.push/{channel_id}", wspush_handler)
@routes.post('/api/rtm.push/{channel_id}')
async def wspush_handler(request):
    channel_id = request.match_info.get("channel_id")
    ch_name = f"ch:{channel_id}"
    data = await request.text()

    conn = await request.app["redis_pool"].acquire()
    await conn.execute("publish", ch_name, data)

    logger.info("#" * 50)
    logger.info(f"Message pushed to {channel_id} data is: {data}")
    logger.info("#" * 50)

    raise aiohttp.web.HTTPOk()


async def listen_to_websocket(ws, app, channel_id):
    # logger.info("listen_to_websocket...begin")
    # logger.info(f'ws in  listen_to_websocket: {ws}')
    try:
        async for msg in ws:
            # 可在此實作從 client 接收訊息時的處理邏輯
            params = json.loads(msg.data)

            model_name = params["model_name"]
            video_path = params["video_path"]
            model_path = params["model_path"]
            output_path = params["output_path"]
            threshold = params['threshold']
            cam = params['cam']
            start_frame = params["start_frame"]
            end_frame = params["end_frame"]
            cuda = params["cuda"]
            chat_id = params["chat_id"]
            message_id = params["message_id"]
            predit_video = params["predit_video"]
            cam_video = params["cam_video"]
            cam_model = params["cam_model"]

            await detector_inference(model_name, video_path, model_path, output_path, threshold, cam, cam_model, predit_video, cam_video, start_frame, end_frame, cuda)

            # if model_name == 'SPPNet':
            #     logger.info("getting start SPPNet")
            #     await dsp_fwa_inference(video_path, model_path, output_path, threshold,
            #                             start_frame=start_frame, end_frame=end_frame, cuda=False)
            #     logger.info("End SPPNet")
            # if model_name == 'XceptionNet':
            #     logger.info("getting start XceptionNet")
            #     await deepfake_detection(video_path, model_path, output_path,
            #                              start_frame=start_frame, end_frame=end_frame, cuda=False)
            #     logger.info("End XceptionNet")
            # if model_name == 'EfficentnetB7':
            #     logger.info("getting start EfficentnetB7")
            #     await dsp_fwa_inference(video_path, model_path, output_path, threshold,
            #                             start_frame=start_frame, end_frame=end_frame, cuda=False)
            #     logger.info("End EfficentnetB7")

            # logger.info("Message received from client--msg.data: ", msg.data)
            # logger.info("Message received from client--type: ", type(json.loads(msg.data)))
            ws_client = app["websockets"][channel_id]
            logger.info(f'ws_client in  websocket: {ws_client}')
            await ws_client.send_str('Done!!')
            logger.info("#" * 50)

    finally:
        return ws


async def listen_to_redis(app, channel_id):
    conn = await app["redis_pool"].acquire()
    ch_name = f"ch:{channel_id}"
    try:
        await conn.execute_pubsub("subscribe", ch_name)
        channel = conn.pubsub_channels[ch_name]

        logger.info(f"Channel created: {ch_name}")

        ws = app["websockets"][channel_id]
        logger.info(f'ws in  redis: {ws}')
        while await channel.wait_message():
            msg = await channel.get(encoding="utf-8")
            logger.info(f'print redis msg:{msg}')
            await ws.send_str(msg)
        await conn.execute_pubsub("unsubscribe", ch_name)
    except Exception as e:
        logger.info(e)
    # except (asyncio.CancelledError, asyncio.TimeoutError):
    #     pass


async def on_startup(app):
    # 建立 Redis connection pool
    address = ('localhost', 6379)
    encoding = 'utf-8'
    app["redis_pool"] = await aioredis.create_pool(
        #"redis://localhost:6379?encoding=utf-8",
        address=address,
        encoding=encoding,
        create_connection_timeout=1.5,
        minsize=1,
        maxsize=1000,
    )


async def on_shutdown(app):
    # 關閉所有 websockets 與 Redis 連線
    await app["redis_pool"].wait_closed()
    for ws in app["websockets"].values():
        await ws.close(code=WSCloseCode.GOING_AWAY, message="Server shutdown")


@aiohttp.web.middleware
async def auth_middleware(request, handler):
    if request.path.startswith("/api/"):
        logger.info(f"check token....for {handler}")
        pass
        # Token 驗證:
        # 驗證失敗:
        # raise aiohttp.web.HTTPForbidden()
    return await handler(request)


def main():
    app = aiohttp.web.Application(middlewares=[auth_middleware])
    
    # 建立一個 reference dict 準備關聯全部 ws 連線物件, key 為 {channel_id}
    app["websockets"] = weakref.WeakValueDictionary()
    app.add_routes(routes)
    #app.router.add_get("/ws/{channel_id}", ws_handler)
    #app.router.add_get("/api/rtm.connect", wsticket_handler)
    #app.router.add_post("/api/rtm.push/{channel_id}", wspush_handler)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    aiohttp.web.run_app(app, host=WS_HOST, port=WS_PORT)


if __name__ == "__main__":
    main()
