# Quick start and user manual (about 10 minutes)

## 0. What this is
This is a miniature version of the full SIH26169 deliverable set:

| PS deliverable | Where it is in this repo |
|---|---|
| Software application (virtual scene, moving beacon, virtual pan/tilt camera, disturbances, real-time statistics) | `gui.py` |
| AI component, trained on your machine | `train.py`, which writes `models/beacon_verifier.joblib` |
| Benchmark-2: `.mp4` input with the camera control bypassed | **Load MP4** button in the GUI, or `app.py video` |
| Performance log (duration, FPS, acquisition, mean/max error, lock retention, re-acquisition, processing time) | written automatically to `results/` |
| Benchmark suite (every motion × every disturbance, plus an ablation) | `bench.py` |
| Source code, modular and commented | `fsoctrack/` |

## 1. Install (about 2 minutes, once)
macOS / Linux:
```bash
bash setup.sh                  # creates .venv inside the project and installs requirements
source .venv/bin/activate      # do this in every new terminal before running anything
```
Windows (PowerShell):
```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
```
The `.venv` folder belongs to the folder it was created in. If you move or rename the project folder, run the setup script again; it deletes the old `.venv` and makes a fresh one. If `python gui.py` ever reports "Could not find the Qt platform plugin", the wrong Python is running: activate the venv (the prompt shows `(.venv)`) and try again.

## 2. Train the AI (about 3 minutes, optional: a trained model is already included)
```bash
python train.py
```
Watch the four stages:
1. **DATA** — about 150 random scenes are simulated, and every bright blob is labelled automatically as beacon or clutter.
2. **TRAIN** — a neural network learns to tell beacons from clutter. The printed training loss should fall.
3. **TEST** — accuracy, precision and recall on blobs the network never saw.
4. **VALIDATE** — closed-loop tracking, Classical vs AI, on hard scenarios.

Run `python train.py --scenes 400` for a bigger model.

## 3. Run the application
```bash
python gui.py
```

| Control | What it does |
|---|---|
| Start / Pause | Runs the 30 Hz loop |
| Reset / new target | New random beacon position and trajectory. The camera starts at the screen centre. |
| Occlude beacon 1 s | Hides the beacon so you can watch COAST → re-acquisition |
| Click on the SCREEN map | Teleports the beacon, forcing a real loss, a search and re-acquisition |
| Preset | One-click benchmark scenarios, e.g. Fog + all noise, Rain + decoys, STRESS |
| Target motion / speed / size / decoys | Line, circle, figure-8, random, spiral, sine; 0.2–4 °/s; 5–20 px; extra beacons |
| Camera FOV, max pan / tilt speed | Default 4° × 3°, 5 °/s |
| Image noise | Gaussian (σ up to 20), salt & pepper (up to 20%), Poisson |
| Weather | Clear, haze, fog, rain, low light |
| Turbulence, jitter, platform motion | Scintillation; jitter up to ±20 px/frame; platform linear/circular/random up to 20 px/frame |
| Detector | AI (neural verifier), Classical, or Naive threshold. Switch live and compare. |
| Save performance report | Writes `results/live_<time>_performance.txt/.json/_frames.csv` |
| Record screen to MP4 | Records the window for your demo video |
| Load MP4 (Benchmark-2) | Tracks a video file. If `<name>_truth.csv` sits next to it, errors are scored. |

Colours: the camera border and state label turn **orange** in SEARCH, **green** in TRACK and **yellow** in COAST. A blue rectangle is the gate window. A pink × is the predicted position.

## 4. Benchmark-2 rehearsal
```bash
python app.py makevideo --out bench_test.mp4 --seconds 10
```
Then click **Load MP4** and pick `bench_test.mp4` (the truth CSV written next to it is picked up automatically, so the errors are scored). Video files are ignored by git, so they never end up in the repository.

## 5. Full benchmark (optional, about 4 minutes)
```bash
python bench.py && python summarize.py && python make_figures.py
```
Results go to `results/RESULTS.md`, `results/benchmark.json` and the figures to `docs/fig_*.png`.

## 6. Standalone executable (optional)
```bash
pip install pyinstaller
pyinstaller --onefile --windowed --add-data "models:models" gui.py
```
On Windows, use `models;models` in the `--add-data` argument.
