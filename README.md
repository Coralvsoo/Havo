# Havo

Webcam app that recognizes registered users, detects recyclable items and bins,
and awards points when a recyclable is dropped in the configured drop zone.
User face embeddings and points are stored in a local SQLite database.

In pretrained YOLO-World mode, the app automatically detects trash cans and
recycling bins, recognizes registered faces, and awards points when recyclable
items are dropped into the detected bin's opening. Fine-tuned and comparison
modes ask you to select a target bin at webcam startup.

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

In fine-tuned or comparison mode, the webcam asks you to select the target bin
for that run. Drag a rectangle around the bin and press Enter; press Esc to
cancel. Only that selected bin is used to position the drop zone. The pretrained
YOLO-World mode does not ask for selection; it automatically prefers a detected
recycling bin, then falls back to a detected trash can.

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
first-dataset model. The app looks for the checkpoint in
`Downloads\P04_bin_dataset2_best.pt`, the project root as
`P04_bin_dataset2_best.pt`, then `weights\p04_bin\best.pt`. It combines the
selected bin detector with pretrained YOLO-World item detection. If no
fine-tuned checkpoint is available, the old YOLO-World model remains available
for both items and bins.

## Local privacy

Face images, the `known_faces` folder, and the SQLite database are local data and
are intentionally excluded from Git. Do not commit real face photos, exported
face embeddings, or other personal data. User registration creates local
embeddings in `smartclean_app.db`.
