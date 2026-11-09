# Havo

Webcam app that recognizes registered users, detects recyclable items and bins,
and awards points when a recyclable is dropped in the configured drop zone.
User face embeddings and points are stored in a local SQLite database.

Point the webcam at a recycling bin: the app detects the bin on each frame and
places the highlighted drop area over its opening. Bottles, cans, and cups are
then tracked relative to that opening; a recognized bin must remain visible for
drop tracking and points to work.

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

The app prompts you to register a face before starting the webcam. Alternatively,
register a user from a local image:

```powershell
python face.py --add-user Name path\to\face.jpg
```

The first webcam run may download the YOLO-World model weights. A working
internet connection is required for that initial download.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Local privacy

Face images, the `known_faces` folder, and the SQLite database are local data and
are intentionally excluded from Git. Do not commit real face photos, exported
face embeddings, or other personal data. User registration creates local
embeddings in `smartclean_app.db`.
