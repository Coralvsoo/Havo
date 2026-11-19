# Havo

Webcam app that recognizes registered users, detects recyclable items and bins,
and awards points when a recyclable is dropped in the configured drop zone.
User face embeddings and points are stored in a local SQLite database.

The webcam app requires registered-face recognition and recyclable detection.
Its `lock` workflow proposes a bin at startup, asks you to accept it or redraw
the rectangle, then follows that same bin with detections on each frame. If the
bin cannot be matched for three consecutive frames, point awards pause until
you press C to recalibrate. A manually drawn bin rectangle remains the target.
Points are based on a recyclable moving into the on-screen drop zone and then
either settling or disappearing for four consecutive frames. The zone extends
from 0.20 bin heights above the bin's top to 0.40 bin heights below it, with a
5% side inset. The camera event is not independent proof that the item
physically landed in the bin.

## Requirements

- Python 3.10 or newer
- A webcam

## Setup

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the app:

```powershell
python face.py
```

The startup window lets you select pretrained YOLO-World, YOLO-World for
recyclable items plus the fine-tuned bin-only YOLOv8 checkpoint, or a side-by-side
live comparison. Comparison mode runs both pipelines on the same webcam frames,
first asks you to drag a rectangle around the one bin to use and press Enter.
It displays per-frame inference time and the number of bin boxes each detector
proposed, then prints mean/median/p95 latency and proposal frame rate when you
quit. The selected rectangle, not those proposals, defines the drop zone. The
proposal rate is not an accuracy measure; inspect missed and false detections.
Face recognition also runs in comparison mode and shows the same recognized
person on both detector panels. A registered face is required; comparison mode
does not award points.

The app prompts you to register a face before starting the webcam. Alternatively,
register a user from a local image:

```powershell
python face.py --add-user Name path\to\face.jpg
```

The app samples 30 frames to find a stable bin and asks you to confirm the
proposed rectangle. Press Enter or Space to accept it, R to redraw, or Esc to
select the bin manually. The detector then follows the accepted bin and updates
the drop zone as its box moves. If it cannot match the bin for three
consecutive frames, the app pauses point awards until you press C to
recalibrate. A manually selected rectangle stays fixed.

Face recognition is required for points. A confirmed drop is assigned only to
a registered user identified near the action; otherwise it is recorded without
awarding points. Video input and session recording are available explicitly:

```powershell
python face.py --video path\to\test.mp4
python face.py --record
python face.py --sensor serial:COM3 --record
python face.py --record-screen
```

Screen recording also starts and stops with V during a session. It saves the
annotated application view and an event-time CSV under `screen_recordings\` by
default; `--record-screen DIR` selects another directory. Unlike `--record`,
the screen recording includes the visible face and identity overlay, so use it
only with the people's consent.

`--record` creates a timestamped folder under `sessions\` with blurred camera
segments, event clips and CSV/JSON metadata. Face blurring is enabled unless
`--no-blur` is specified. `--max-storage-gb` limits storage by removing the
oldest segments while preserving clips and metadata. `--sensor mock` selects a
test source; `--sensor serial:PORT` connects to the ESP32 serial sensor. When a
sensor is connected, its events are the independent ground truth for the live
drop self-check. Without a sensor, use the saved clips for human review.

Review clips and compare all three zone modes against sensor events or human
labels:

```powershell
python scripts\review_session.py sessions\<timestamp>
python scripts\evaluate_session.py sessions\<timestamp>
python scripts\evaluate_session.py --sessions sessions\run1 sessions\run2
```

The review window loops each event clip. Press Y if a real drop happened, N if
not, S to leave it unlabeled, or B to revisit the previous clip. For a
`zone_activity` clip, Y means the app missed a real drop.

The first webcam run may download the YOLO-World model weights. A working
internet connection is required for that initial download.

## FastAPI service

The API accepts uploaded webcam frames; a hosted API cannot open a webcam
connected to your computer directly. It uses the same database as the desktop
app when both run on the same machine with the same `HAVO_DATABASE_PATH`.
A deployed API has its own database; local desktop users are not synced
automatically.

Start the API locally from PowerShell:

```powershell
$env:HAVO_API_KEY = "replace-with-a-long-random-secret"
.\env\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) and use **Authorize**
to enter the `X-API-Key`. `GET /health` is public; face registration, frame
processing, point lookup, and session reset require the key. Frame processing
recognizes the registered face, detects recyclable items and bins, tracks drops
between frames sharing a `session_id`, and awards points after a confirmed drop.
The API returns JSON detections and points; it does not return an annotated
image.

For Render, create a Python web service for this repository with build command
`pip install -r requirements.txt` and start command
`uvicorn api:app --host 0.0.0.0 --port $PORT`. Set `HAVO_API_KEY` as a secret
environment variable in Render; never commit the key. The deployed Swagger UI
will be at `https://<your-service>.onrender.com/docs`.

The default SQLite database and in-memory drop-tracking sessions are local to
one service instance. Cloud deployments need persistent disk storage configured
with `HAVO_DATABASE_PATH` for registrations and points to survive redeploys;
drop sessions reset when the service restarts and require requests to reach the
same instance. Face embeddings are biometric data, so use HTTPS, protect the API
key, and do not upload real face images to a public service unless you are
comfortable with the hosting and data-retention setup. Uploaded registration
images are processed in memory and are not saved.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Fine-tuning the recycling detector

Open `P04_bin_experiment.ipynb` in Google Colab or Jupyter to fine-tune a
separate YOLOv8n detector on the second Roboflow bin dataset. Set the ZIP path
in the notebook and optionally upload the first dataset's `P04_bin_best.pt` to
compare both trained models on the second dataset's held-out test images. The
notebook preserves the dataset's train/validation/test split and performs
five-fold cross-validation only on train+validation data. The existing
pretrained YOLO-World model continues to detect `bottle`, `can`, and `cup`.
Image-level accuracy and ROC-AUC are labeled separately from box-localization
metrics.

After running the second-dataset notebook, it exports
`P04_bin_dataset2_best.pt`. Keep this separate while comparing it with the
first-dataset model. The app looks for the checkpoint in the project root as
`P04_bin_dataset2_best.pt`, then `Downloads\P04_bin_dataset2_best.pt`, then
`weights\p04_bin\best.pt`. Fine-tuned mode combines this bin detector with
pretrained YOLO-World item detection. The pretrained-only selection remains
available for items and bins.

## Local privacy

Face images, the `known_faces` folder, and the SQLite database are local data and
are intentionally excluded from Git. Do not commit real face photos, exported
face embeddings, or other personal data. User registration creates local
embeddings in `smartclean_app.db`.
