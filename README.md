# Self-Driving Car: Behavioral Cloning

A CNN that learns to steer a car in the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim) from front-camera images (PilotNet-style, Keras).

**Demo video:** ADD_YOUR_VIDEO_LINK_HERE

## How it works
1. Drive the lake track in the simulator's Training Mode and record (center, left and right camera images + steering).
2. Train a CNN to predict steering from the camera image.
3. `drive.py` connects to the simulator in Autonomous Mode and sends the model's steering back every frame.

## Training details
- Data: about 6,100 frames recorded on the lake track, including recovery driving.
- Labels: rolling-mean smoothing (keyboard steering is step-like), left/right camera offset of 0.1.
- Split: interleaved 200-frame blocks, so validation includes turns. Balancing on train only.
- Preprocessing: crop sky and hood, resize to 200x66, convert to YUV.
- Augmentation: horizontal flip (steering negated) and brightness.
- Model: 5 conv layers + dense head, MSE loss, Adam, EarlyStopping.

## Results (validation)
| Metric | Model | Predict-0 baseline |
|---|---|---|
| Turn-frame MAE (\|steering\| > 0.05) | 0.071 | 0.140 |
| R2 | 0.14 | - |

R2 is low because about 70% of frames are straight (steering 0). The turn-frame MAE and the simulator run are the more meaningful checks.

## Run it
Use Python 3.11 (the pinned socketio and eventlet versions need it).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python drive.py model.keras
```
Then open the simulator (Term 1, Version 2), pick the **Lake** track and click **Autonomous Mode**.

`drive.py` options: `TARGET_SPEED`, `STEER_GAIN`, and a stuck-recovery routine that reverses when the car stops.

## Files
- `self-driving-car-fixed.ipynb`: training notebook (Kaggle)
- `drive.py`: simulator client
- `model.keras`: trained model
- `config.json`: input size and settings
- `requirements.txt`: dependencies

## Notes
The dataset is not included. Record your own in the simulator's Training Mode and set `DATA_DIR` in the notebook.
