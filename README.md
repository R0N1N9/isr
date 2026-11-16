# ISR tools for the DJI Mavic 3

Offline field and studio tools for a Mavic 3 + DJI RC:

1. **`isr triage`**: scans every photo on an SD card or folder for possible people and person-related
   objects (clothing, packs, tarps, tents, vehicles). It ranks the hits, places each one on the map,
   and produces a review page plus GPX/KML/CSV for ground teams.
2. **`isr plan`**: generates photogrammetry mapping grids and SAR search patterns (parallel track,
   expanding square, sector) as DJI waypoint `.kmz` files, plus KML/GPX/GeoJSON for CalTopo or
   Google Earth.
3. **`isr studio`**: ranks every photo on a card for portfolio or print potential, groups bursts and
   re-takes so each moment shows its best frame, and auto-edits the top picks. You choose favourites
   in a gallery with a before/after slider.
4. **`isr enhance`**: light, professional automatic touch-ups (white balance, dehaze, tone, clarity,
   vibrance, horizon straightening, sharpening) written to new files with all metadata kept.

Everything runs on a laptop with no internet connection. Nothing is ever written to the SD card.

![Studio gallery](docs/studio_gallery.jpg)

---

## Install (once, with internet)

Needs Python 3.10+.

```bash
git clone <this repo> isr && cd isr
python -m venv .venv
```

Activate the environment. On macOS / Linux:

```bash
source .venv/bin/activate
```

On Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Then install:

```bash
pip install -e .
```

Optional AI detector (large download, about 1–2 GB with PyTorch):

```bash
pip install -e ".[ai]"
```

Fetch the model weights while you still have internet. Run this once; it creates `yolov8n.pt`:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

---

## Photo triage

```bash
isr triage /Volumes/DJI_SD/DCIM --open
```

On Windows, with the card at `E:`:

```bash
isr triage E:\DCIM --open
```

With the missing person's clothing colours:

```bash
isr triage E:\DCIM -c red -c blue -o C:\SAR\incident42 --open
```

The tool searches the folder recursively, reads each photo's GPS position, altitude and gimbal angle,
runs the detectors, and writes the following to the output folder (default `./triage_<timestamp>`):

| File | Use |
|---|---|
| `report.html` | Review page: map of photo coverage and hits, ranked candidate cards. Works offline; copy the whole folder to a USB stick and it still works. |
| `detections.gpx` | Waypoints for handheld GPS units |
| `detections.kml` | Google Earth / CalTopo / ATAK (includes photo footprints, hidden by default) |
| `detections.geojson` | CalTopo / SARTopo import, QGIS |
| `detections.csv` | Incident log / spreadsheet |
| `crops/`, `previews/` | Zoomed crops and annotated photos |

![Triage review page](docs/triage_report_example.jpg)

### Reviewing

Open `report.html`. Candidates are sorted by score.

| Key | Action |
|---|---|
| `J` / `K` (or arrow keys) | Next / previous candidate |
| `Space` | Full annotated photo |
| `C` | Confirm |
| `X` | Dismiss |
| `U` | Undo |

Confirmed and dismissed marks are kept in that browser. **Export confirmed GPX** gives you only the
finds you've confirmed, ready to send to a ground team. Click a coordinate to copy it.

### Detectors

| Detector | What it finds | Flag |
|---|---|---|
| **Colour anomaly** (default) | Anything statistically unusual for both the whole photo *and* its immediate ~8 m surroundings. Catches bright clothing, tarps, gear and vehicles, and needs no description of the person. | on by default; `--sensitivity 1.5` flags more, `0.7` fewer; `--no-anomaly` turns it off |
| **Target colour** | Known clothing or gear colours that also stand out from their surroundings (so a field of dry grass doesn't match "orange"). Hits found by both detectors rank highest. | `-c red -c "#1e90ff"`; names: red, orange, yellow, blue, purple, pink, magenta, cyan, white, green (green floods on vegetation) |
| **AI (YOLO)** | People and objects, run on overlapping full-resolution tiles so small people stay visible to the model | `--ai`, `--ai-model your_model.pt` |

Size filtering uses each photo's real scale (cm per pixel from altitude). Blobs smaller than
0.04 m² or larger than 12 m² are ignored (`--min-area` / `--max-area`), as are long thin
features such as fences and road markings.

**About the AI option:** stock YOLO weights are trained on ground-level photos and do poorly
looking straight down (in testing they called playground equipment a "dog" and missed parked cars).
Treat them as a supplement. A model fine-tuned on aerial SAR imagery (for example one trained on
the HERIDAL dataset) can be passed with `--ai-model`, and every class it knows is treated as relevant.
On CPU it adds about 5–10 s per photo.

### Accuracy and limits

* **Position** comes from the drone's GPS, its altitude above the take-off point and the gimbal
  angle, projected onto flat ground. Straight-down photos are typically within a few metres. Oblique
  photos, and terrain above or below take-off, are worse. If the search area sits 30 m below
  where you launched, pass `--ground-offset -30`. Cards say "drone pos" when a hit couldn't be
  projected and the drone's own position is shown instead.
* **It narrows the search; it does not clear an area.** Canopy, shadow, and drab or camouflage
  clothing all hide people from colour-based detection. Someone still needs to look at the photos
  of high-probability areas.
* JPEGs only. If you shoot JPEG+DNG, the DNGs are ignored.

### Flying for triage

Use `isr plan calc` to choose an altitude. The table shows how many pixels cover a person's
shoulders at each height:

```
 alt m  GSD cm/px    footprint m  person px*
    40       1.07     56 x 42             47
    60       1.60     84 x 63             31
    80       2.13    113 x 85             23
   100       2.67    141 x 106            19
```

**40–70 m with the camera straight down** is a good balance for people: they stay 25–50 px across
while each photo still covers a useful area. Keep the gimbal at −90° where you can. Oblique shots
cover more ground but geolocate worse and hide people behind vegetation.

---

## Mission planning

```bash
# Altitude / resolution / footprint table
isr plan calc

# Photogrammetry grid at 2.5 cm/px over a rotated 400x250 m rectangle
isr plan grid --center 39.7000,-105.2000 --size 400x250 --rotation 20 --gsd 2.5

# Same over a polygon drawn in CalTopo / Google Earth (KML, GeoJSON or GPX)
isr plan grid --aoi search_area.kml --alt 80 --front 80 --side 70

# 3D model: second grid at 90 degrees
isr plan grid --aoi site.kml --alt 70 --crosshatch

# SAR parallel-track photo search (default 60 m, 30 % / 20 % overlap)
isr plan search --aoi segment_A.kml --alt 50

# SAR expanding square / sector search around the last known point
isr plan square --datum 39.7012,-105.2034 --radius 300 --alt 50
isr plan sector --datum 39.7012,-105.2034 --radius 200 --heading 270
```

Each mission is written to `./missions/` as `.kmz` (DJI waypoint format), `.kml` / `.geojson` /
`.gpx` (to check in CalTopo or Google Earth), and `.txt` (summary).

The summary shows altitude, image resolution, footprint, spacings, photo count, path length,
estimated flight time and batteries (planned at 25 min each). Missions over 200 waypoints are split
into parts (`--max-waypoints`).

The aircraft flies to each waypoint, stops, points the camera down and takes a photo. Stopping to
shoot avoids smearing, because the consumer Mavic 3 has no mechanical shutter.

### Loading a mission into DJI Fly (unofficial)

The consumer Mavic 3 isn't supported by DJI's developer kit, so third-party apps can't fly it.
DJI Fly can't import missions either. The method below is a community workaround: you create a
mission in DJI Fly, then replace its file with one generated here.

* It isn't supported by DJI and may stop working after a firmware or app update.
* Folder names may differ between DJI Fly versions.
* **Always check the route on the DJI Fly map before take-off, and start with a small test mission.**

1. In DJI Fly, create a simple waypoint mission near the site (two or three points) and save it.
2. Connect the DJI RC to the laptop with USB-C and open its storage. Missions live under
   `Android/data/dji.go.v5/files/waypoint/<ID>/<ID>.kmz`. Find the newest one.
3. Copy that `.kmz` to the laptop. Generate your mission using it as the template, so the drone
   and camera IDs match your aircraft:
   ```bash
   isr plan grid --aoi site.kml --gsd 2.5 --template <ID>.kmz
   ```
4. Rename the generated `.kmz` to exactly `<ID>.kmz` and copy it over the original on the RC. Keep a
   backup of the original.
5. Restart DJI Fly, open the mission, and check the route, altitude and photo actions on the map.

Without `--template` the file still validates, but it uses DJI's published Mavic 3 *Enterprise*
drone ID, which DJI Fly on a consumer Mavic 3 may reject.

### Processing mapping photos

* **WebODM / OpenDroneMap** (free) handles Mavic 3 photos well.
* **Agisoft Metashape** and **Pix4D** are the paid alternatives.

Without RTK, measurements within a map will be good, but its absolute position is only accurate to a
few metres. Add ground control points if you need survey-grade placement.

---

## Studio: find your portfolio shots

```bash
isr studio E:\DCIM --open
```

On macOS or Linux, with a different number of suggestions:

```bash
isr studio /Volumes/DJI_SD/DCIM --top 40 --open
```

Studio reads every JPEG on the card and writes to `./studio_<timestamp>/` (or `-o FOLDER`):

| Output | What it is |
|---|---|
| `gallery.html` | The gallery. Works offline and defaults to dark mode (the standard for judging photos). |
| `edited/` | Full-resolution auto-edits of the top picks, with original EXIF, GPS, DJI metadata and colour profile kept |
| `thumbs/`, `previews/` | Gallery images |
| `studio.json` | All scores, for your own tooling |

### How photos are ranked

Taste is subjective, so Studio doesn't pretend to have it. Instead it measures what reliably
separates keepers from rejects in aerial work. Each photo gets a 0–100 score from two halves:

| Technical (sellable at all?) | Artistic (does it have what aerial art is known for?) |
|---|---|
| **Sharpness**: motion blur and missed focus, with sensor grain discounted so noise isn't mistaken for detail | **Colour**: vividness, with garish extremes penalised |
| **Exposure**: mid-tone placement, clipped highlights, crushed shadows | **Composition**: a clear subject on a third or dead centre, or strong mirror symmetry |
| **Contrast**: flat, hazy frames score low | **Simplicity**: large calm areas of water, sand, snow or fields (minimalism sells) |
| **Noise** | **Pattern**: repeating structure such as crop rows, solar farms or waves |

A photo's strongest artistic quality carries extra weight: a great abstract pattern shot doesn't
also need to be colourful. A blurry or badly exposed frame can never outrank a clean one. Each
photo gets plain-language tags (*Tack sharp, Vivid colour, Minimal, Symmetry, Pattern*) and
warnings (*Soft / motion blur, Clipped highlights, Flat / hazy, Horizon tilted 2.5°*).

**Similar frames** (bursts, brackets, hovering re-takes taken within 3 minutes) are grouped as one
*moment*, and only the best frame competes for the top picks. The others appear under "Similar
frames" in the viewer. Use `--similarity` to group more or less aggressively.

In testing on real aerial photos, blurred, motion-blurred, over-exposed, under-exposed, hazy and
noisy copies ranked below their originals in every case. Treat the ranking as a fast first cull;
the final call is yours.

### Using the gallery

| Control | Action |
|---|---|
| **Suggested** / **Best of each** / **All photos** / **Picks** / **Rejected** | Filter what's shown |
| **Sort** | Overall score, time taken, sharpness, colour or composition |
| `P` (or the heart) | Pick a photo |
| `X` | Reject a photo |
| `←` `→` | Previous / next |
| `Space` | Open the viewer |
| `E` | In the viewer, flip between original and edited |
| Drag on the photo | Before/after slider |

Picks are remembered in that browser. **Export picks** downloads `picks.txt`; run the command shown
there to make full-resolution edits of exactly your picks:

```bash
isr enhance --list picks.txt -o portfolio
```

![Studio viewer with before/after slider](docs/studio_viewer.jpg)

---

## Enhance: automatic touch-ups

```bash
isr enhance DJI_0412.JPG DJI_0418.JPG -o edited
isr enhance E:\DCIM\DJI_001 -o edited --strength 0.4
isr enhance --list picks.txt -o portfolio
```

The aim is the first five minutes of a professional edit, not a filter or a "look". Every step is
measured from the photo itself, so a well-exposed frame is barely touched:

1. **White balance**: partial correction of colour casts, measured only from genuinely grey surfaces
   (roads, roofs, rock, cloud). Left alone when there are none, so green landscapes aren't turned magenta.
2. **Dehaze and levels**: removes the haze veil common at altitude and sets the white point, without
   clipping.
3. **Tone**: gently lifts dark frames or pulls down bright ones.
4. **Contrast**: an adaptive S-curve; flat frames get more, punchy frames almost none.
5. **Clarity**: mid-tone local contrast, which brings out terrain texture.
6. **Vibrance**: boosts muted colours more than already-saturated ones.
7. **Straighten**: levels a tilted horizon on oblique shots, then crops to the largest clean
   rectangle. Only when a horizon is detected with confidence, never on top-down shots.
8. **Sharpen**: fine output sharpening sized to the image.

`--strength` runs from 0 (nothing) to 1 (strong); the default is 0.6. `--no-straighten` turns
rotation off. Output is a quality-95 JPEG with full-resolution colour (4:4:4) and the original
EXIF, GPS, DJI XMP and colour profile, so edited files still geotag correctly.

A 20 MP Mavic 3 frame takes about 5 seconds and around 600 MB of memory.

These are JPEG edits. For the most demanding prints, shoot JPEG+DNG and finish your best frames
from the DNG in Lightroom or Capture One. Studio's ranking is still the fast way to find them.

---

## Other commands

```bash
isr info photo.JPG       # show what the tools read from a photo (GPS, altitude, gimbal, scale)
isr triage --help
isr plan grid --help
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests generate synthetic DJI photos with real EXIF and XMP metadata and known targets
(`tests/synth.py`). They check detection, geolocation to within 1.5 m, exports, pattern coverage,
the `.kmz` structure, Studio's ranking of degraded photos, duplicate grouping, horizon detection,
and that edits keep metadata and never touch the originals.

## Safety and legal

These tools are for planning and for analysing imagery after the flight. You remain responsible for
airspace authorisation, staying under 120 m / 400 ft (the planner refuses anything higher unless
you pass `--allow-high`), keeping the aircraft in sight, and coordinating with whoever is running the
search. In the US, emergency operations outside Part 107 limits go through the FAA's SGI process,
coordinated with the responsible agency.
