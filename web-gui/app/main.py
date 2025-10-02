import json
import logging
import random
import sys
import math
from datetime import datetime, timedelta
from typing import Iterator
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette import EventSourceResponse
from datetime import datetime
import websockets
import time
from ML.interface import TSAIModel
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                   format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Dashboard")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

data_buffer = {
    "bpm": {
        "real": [],
        "predicted": []
    },
    "uterus": {
        "real": [],
        "predicted": []
    }
}
predicted_stacks = {"bpm":[], "uterus":[]}
maxrealdatas = {"bpm":100, "uterus":100}
maxpredicteddatas = {"bpm":100, "uterus":100}
delays_max_counts = {"bpm":50, "uterus":50}
points_before_predicts = {"bpm": 50, "uterus": 50}
predict_points = {"bpm": 10, "uterus": 10}
all_delays = {"bpm":0, "uterus":0}
predict_models = {"bpm" : TSAIModel("./ML/model/patchTST_bpm_50.pt"), "uterus" : TSAIModel("./ML/model/patchTST_uterus_50.pt")}
predict_buffer_array = {"bpm" : [], "uterus" : []}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

async def predict_many(chartid, points, point_count):
    """points = [3, 5, 6, 7...]"""
    if len(points) != 50:
        raise Exception('Invalid predict data')
    points = points[:]

    for i in range(point_count):
        new_point = await predict_one(chartid, points)
        yield new_point
        points.append(new_point)
        points.pop(0)
    yield None

async def predict_one(chartid, points):
    if len(points) != 50:
        raise Exception('Invalid predict data')
    tensor = torch.tensor(points).unsqueeze(0).unsqueeze(2)
    yield predict_models[chartid](tensor).to_list()[0,0,0]


async def get_data_pair(val, chart_id, timestamp, predict_delay = 0, real_delay = 0, predict = False, real = True, timestamp_real = None):
        if timestamp_real is None:
            timestamp_real = timestamp
        
        real_value = val
        
        buffer = data_buffer[chart_id]

        if real:
            real_point = {
                "timestamp": timestamp_real + real_delay,
                "value": round(real_value, 1),
                "type": "real",
                "chart": chart_id
            }

            if len(buffer["real"]) >= maxrealdatas[chart_id]:
                buffer["real"].pop(0)
            buffer["real"].append(real_point)

            if (len(predict_buffer_array[chart_id]) > 0):
                predict_buffer_array[chart_id][0] = real_point 
                print('Change first point on real')

        if predict:
            if (len(predict_buffer_array[chart_id]) <= 0):
                predict_buffer_array[chart_id] = buffer["real"][len(buffer['real'])-50:]
                print(f'Add {len(predict_buffer_array[chart_id])} to buffer')

            predicted_value = await predict_one(chart_id, predict_buffer_array[chart_id])
            predict_buffer_array[chart_id].pop(0)
            predict_buffer_array[chart_id].append(predicted_value)

            predicted_point = {
                "timestamp": timestamp + predict_delay,
                "value": round(predicted_value, 1),
                "type": "predicted",
                "chart": chart_id
            }
            
            if len(buffer["predicted"]) >= maxpredicteddatas[chart_id]:
                buffer["predicted"].pop(0)
            buffer["predicted"].append(predicted_point)
        
        if real and predict:
            data_pair = {
                "type" : 'point',
                "real": real_point,
                "predicted": predicted_point,
                "chart": chart_id,
            }
        elif real:
            data_pair = {
                "type" : 'point',
                "real": real_point,
                "chart": chart_id,
            }
        elif predict:
            data_pair = {
                "type" : 'point',
                "predicted": predicted_point,
                "chart": chart_id,
            }

        yield data_pair

async def generate_chart_data(chart_id: str):
    uri = "ws://ws-server:8765"
    async with websockets.connect(uri) as ws:
        command = {"action": "start", "patient_type" : 'regular', "folder_number" : 5}
        await ws.send(json.dumps(command))
        type = 'bpm' if chart_id == "bpm" else 'uc'
        point_counter = 0
        start_timestamp = time.time()
        last_time_stamp = -1
        predicted_stack = predicted_stacks[chart_id]
        maxrealdata = maxrealdatas[chart_id]
        maxpredicteddata = maxpredicteddatas[chart_id]
        delays_max_count = delays_max_counts[chart_id]
        points_before_predict = points_before_predicts[chart_id]
        predict_point = predict_points[chart_id]
        all_delay = all_delays[chart_id]
        delays = []
        while True:

            msg = await ws.recv()
            data = json.loads(msg)
            print(data)

            if 'type' in data and data['type'] == (type):
                
                val = data[type]

                timestamp = time.time()
                
                if last_time_stamp != -1:
                    delays.append(timestamp-last_time_stamp)
                    if len(delays) > delays_max_count:
                        delays.pop(0)

                if len(delays) > 0:
                    delay = sum(delays)/len(delays)
                else:
                    delay = 0

                if point_counter > points_before_predict and points_before_predict != -1:
                    points = []
                    all_delay = 0
                    for i in range(predict_point):
                        data_pair = await get_data_pair(val, chart_id, timestamp, predict_delay = all_delay, predict = True, real = False)
                        predicted_stack.append(timestamp + all_delay)
                        points.append(data_pair)
                        all_delay += delay
                    shift_data = {"type" : 'shift',
                                "shift": all_delay}
                    points.append(shift_data)
                    predicted_stack = list(map(lambda x : (x-all_delay), predicted_stack))
                    yield f"{json.dumps(points)}\n\n"
                    points_before_predict = -1
                elif points_before_predict == -1:
                    data_pair = await get_data_pair(val, chart_id, timestamp, real_delay=(-all_delay), predict_delay=delay, predict= True, real= True)
                    predicted_stack.pop(0)
                    predicted_stack.append(timestamp)

                    yield f"{json.dumps(data_pair)}\n\n"
                else:
                    data_pair = await get_data_pair(val, chart_id, timestamp, predict = False, real = True)
                    
                    yield f"{json.dumps(data_pair)}\n\n"
                point_counter += 1
                last_time_stamp = timestamp

@app.get("/stream/{chart_id}")
async def stream_data(chart_id: str):
    return EventSourceResponse(generate_chart_data(chart_id))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)