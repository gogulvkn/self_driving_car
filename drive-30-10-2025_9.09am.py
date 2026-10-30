# Run: python drive.py model.keras   (then start the simulator in Autonomous mode)
import sys, json, base64, os
from io import BytesIO
import numpy as np, cv2, eventlet, socketio
from flask import Flask
from PIL import Image
from tensorflow import keras

MODEL_PATH   = sys.argv[1] if len(sys.argv) > 1 else "model.keras"
TARGET_SPEED = 8.0                       # mph, raise slowly once it drives well
STEER_GAIN   = 1.5                       # smoothed labels are small: scale up (try 1.3 to 2.0)
cfg          = json.load(open("config.json")) if os.path.exists("config.json") else {}
H, W         = cfg.get("H", 66), cfg.get("W", 200)
SPEED_MAX    = cfg.get("SPEED_MAX")      # only needed for the speed model

model = keras.models.load_model(MODEL_PATH)
uses_speed = "speed" in [i.name.split(":")[0] for i in model.inputs] or len(model.inputs) == 2

def preprocess(rgb):                     # identical to training
    x = cv2.resize(rgb[60:-25, :, :], (W, H))
    return cv2.cvtColor(x, cv2.COLOR_RGB2YUV).astype(np.float32)

sio = socketio.Server()
app = Flask(__name__)

@sio.on("telemetry")
def telemetry(sid, data):
    if not data:
        sio.emit("manual", data={}, skip_sid=True); return
    speed = float(data["speed"])
    rgb   = np.asarray(Image.open(BytesIO(base64.b64decode(data["image"]))).convert("RGB"))
    x     = preprocess(rgb)[None]
    if uses_speed:
        inp = {"image": x, "speed": np.array([[speed / SPEED_MAX]], np.float32)}
    else:
        inp = x
    steer    = float(np.clip(model.predict(inp, verbose=0)[0, 0] * STEER_GAIN, -1, 1))
    throttle = float(np.clip((TARGET_SPEED - speed) * 0.1, -1, 1))
    print(f"steer {steer:+.3f}  throttle {throttle:+.2f}  speed {speed:.1f}")
    sio.emit("steer", data={"steering_angle": str(steer), "throttle": str(throttle)}, skip_sid=True)

@sio.on("connect")
def connect(sid, environ):
    print("connected", sid)
    sio.emit("steer", data={"steering_angle": "0", "throttle": "0"}, skip_sid=True)

if __name__ == "__main__":
    eventlet.wsgi.server(eventlet.listen(("", 4567)), socketio.WSGIApp(sio, app))
