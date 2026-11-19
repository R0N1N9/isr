# ISRStudio

Offline tools that turn a card of drone (UAV) photos into something you can act on. One app, three
sections, and an art gallery they fill as a by-product:

| Section | For | You get |
|---|---|---|
| **SAR** | Search and rescue: finding a missing person or a target of interest | Every photo scanned, hits ranked and placed on the map, a search report with coordinates, GPX/KML/CSV for ground teams |
| **Survey** | Assessing the security posture of a site | Sightlines, blind spots, camera placement, access points and perimeter, as a report for the client |
| **Stakeout** | A fixed camera left on a gate or approach for a day or a week | When the place is busy, dead, or visited out of hours, and what that means for posts and patrols |
| **Art** (gallery) | Stock libraries, prints and portfolios | Fills itself: stock-worthy frames found in any job's photos, lightly edited, ready to export |

Everything runs on a laptop with no internet connection. Nothing is uploaded, and nothing is ever
written to the SD card.

![ISRStudio](docs/isr_studio.jpg)

## Using the app

```bash
isrstudio
```

That opens ISRStudio in your browser (it is a page served from your own machine at `127.0.0.1`).
Pick a section, press **New**, choose the folder of photos (the card's `DCIM` folder, or any folder
of JPEGs), and wait while it works. You can also drag a folder or an external drive from your file
manager onto a section, or onto the New dialog, instead of browsing to it. Each job then opens in its own page:

* **SAR** opens the review page. Confirm or dismiss candidates, then **Search report** writes a
  printable report of the confirmed finds.
* **Survey** opens the plan editor. Trace the site, then **Generate report**.
* **Stakeout** opens the activity report. Attach it to a survey to add timing to the guard plan.

**Art** is not something you start. After each job the app looks through that job's photos in the
background, keeps the frames that would stand up as pictures (best of each moment, scoring 66 or
more out of 100, with no fault an edit cannot fix), gives them the light automatic touch-up, and
adds them to the Art gallery at the bottom of the home screen. Lookalikes are pruned to the single
best copy: frames from one burst, the same view shot again later, and the same picture arriving
from two different jobs. A stakeout contributes one frame at
most, since a fixed camera takes the same picture all week. Hover a picture to remove it.
**Export all to a folder** copies the full-size edited pictures, with their original GPS and camera
metadata, into `~/ISR-Studio/art-export/<date>/` together with `pictures.csv`, ready to upload to
a stock library. Check each one first for recognisable people or private property: stock libraries
ask for a release.

No photos to hand? Each section has **Try the demo**, which paints a made-up set and runs the real
tools on it: a six-frame search leg with three people to find (SAR), a property with a hidden side
door (Survey), and a week of a trail camera on a gate with rush hours, quiet weekends and visits in
the small hours (Stakeout).

Jobs are kept in `~/ISR-Studio/<section>/<job>/` and listed on the home screen, so you can reopen
or delete them. Each job folder holds its report and exports and can be copied to a USB stick.
**Quit** on the home screen stops the app.

On Linux, add it to the applications menu so it starts with a click instead of a terminal:

```bash
isrstudio --install-launcher
```

Every section also works from the command line, which is what the rest of this document describes:
`isr triage` (SAR), `survey` (Survey), `isr art` to rank a whole card for art yourself and
`isr enhance` for the touch-ups, plus `isr plan` for
mapping grids and SAR search patterns.

### Which aircraft

Any drone that writes JPEGs with GPS works for **Art**, and its photos can be placed on a map.
**SAR** geolocation and the **Survey** base map also need the camera angle and the height above
take-off, which the tools read from DJI-format photo metadata (the `drone-dji` XMP block). Photos
without it are still scanned: SAR hits are then given at the aircraft's position rather than
projected onto the ground, and Survey treats the photos as ground-level shots. Phone photos work in
Survey as walk-round evidence. The mission planner writes DJI waypoint files.

---

## Install (once, with internet)

Needs Python 3.10+.

```bash
git clone https://github.com/r0n1n9/isrstudio.git && cd isrstudio
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

* **WebODM / OpenDroneMap** (free) handles these photos well.
* **Agisoft Metashape** and **Pix4D** are the paid alternatives.

Without RTK, measurements within a map will be good, but its absolute position is only accurate to a
few metres. Add ground control points if you need survey-grade placement.

---

## Art: find your portfolio and stock shots

```bash
isr art E:\DCIM --open
```

On macOS or Linux, with a different number of suggestions:

```bash
isr art /Volumes/DJI_SD/DCIM --top 40 --open
```

Art reads every JPEG on the card and writes to `./studio_<timestamp>/` (or `-o FOLDER`):

| Output | What it is |
|---|---|
| `gallery.html` | The gallery. Works offline and defaults to dark mode (the standard for judging photos). |
| `edited/` | Full-resolution auto-edits of the top picks, with original EXIF, GPS, DJI metadata and colour profile kept |
| `thumbs/`, `previews/` | Gallery images |
| `studio.json` | All scores, for your own tooling |

### How photos are ranked

Taste is subjective, so Art doesn't pretend to have it. Instead it measures what reliably
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

![Art viewer with before/after slider](docs/studio_viewer.jpg)

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

A 20 MP frame takes about 5 seconds and around 600 MB of memory.

These are JPEG edits. For the most demanding prints, shoot JPEG+DNG and finish your best frames
from the DNG in Lightroom or Capture One. Art's ranking is still the fast way to find them.

---

## Survey: site security assessment

The Survey section of ISRStudio. Also its own command, `survey` (the same thing as `isr site`).

For a property you own or are responsible for: where can someone approach unseen, which doors and
gates are unwatched, and where would cameras do the most good.

```bash
survey demo --edit               # a made-up property, to see how it works
survey new E:\DCIM -o my_site --name "12 Mill Lane"
```

![Site editor](docs/site_editor.jpg)

### 1. Capture

No mapping mission or automated flight is needed. Fly by hand and take ordinary photos:

| Shot | How | What it gives |
|---|---|---|
| **One from above** (best) | Climb to 50–100 m over the middle of the property, tilt the gimbal fully down (−90°), take one photo that includes the street outside. | A clean, to-scale, north-up base map to draw on |
| **A lap around the building** | Circle at 30–50 m with the camera tilted 40–60° down, a photo every quarter turn or so. | Fills in the base map if there is no shot from above; shows walls, doors and what is under eaves and trees; attached as evidence in the report |
| **Phone photos on foot** (optional) | Location on. Each door, gate, side passage and fence line, and the view in from the street. | Placed on the map with the direction the camera was pointing |

Put all the JPEGs in one folder. The survey works out which is which from the gimbal angle and
height recorded in each photo.

With **only angled shots**, the map is still built: each patch of ground is taken from the photo that
looks at it most steeply. Ground-level detail lands in the right place, but anything tall leans away
from the drone by roughly its own height, so trace buildings where the walls meet the ground. Photos
tilted less than 35° below the horizon show too little ground and are used as evidence only.
With **phone photos only** there is no aerial image, so you sketch the plan on a 10 m grid using the
photo positions as a guide.

### A 3-D map (optional)

The quick base map above is a few photos flattened onto the ground. A proper **photogrammetry**
run turns many overlapping photos into a true orthophoto and a model of the height of everything on
the site. ISRStudio does not do that itself; it uses [OpenDroneMap](https://opendronemap.org), the
standard free and open-source tool, in one of two ways:

* **Use a result you already have.** Process the photos in WebODM (or ODM), then start the survey
  from that project folder, or pass it with `survey new PHOTOS --map ODM_PROJECT`. A project that
  sits inside the photo folder is found automatically.
* **Build it from the app.** Tick *Build a 3-D map* in the New survey dialog, or use
  `survey new PHOTOS --build-map`. This needs Docker, an internet connection the first time (the
  OpenDroneMap image is several GB), and patience: allow tens of minutes to an hour and 8 GB of
  memory or more. It is too much for an old laptop; run it on a bigger machine and copy the
  project folder across.

To be worth reconstructing, the photos must overlap heavily: fly a grid, or two laps at different
heights, with 70% overlap or more, straight down and at an angle. Twenty photos is a minimum.

With a 3-D map the survey changes in three ways:

* **Measured heights.** Anything the reconstruction shows standing 0.6 m or more above the ground
  blocks sight at its real height, whether or not you traced it. You still draw buildings, fences
  and doors (the tool needs to know what things *are*), but an untraced shed, a van or an overgrown
  corner is no longer invisible to the analysis. Each drawn item shows its measured height with a
  **Use** button.
* **Two assessments.** The report gives the headline figures both ways: from the plan as drawn and
  from the measured heights. Untick *Use measured heights* under **Site** to work from the plan alone.
* **3-D view.** The **3D** button in the editor shows the site as a model you can turn and zoom,
  with the aerial image draped over it and cameras, access points, blind spots, guard posts, the
  patrol route and watch positions standing on it. (It works without a reconstruction too, using
  the heights you typed.)

Limits: a reconstruction is a surface seen from above. It cannot see under a tree canopy or a
porch, so ground beneath them counts as hidden; thin fences and railings often do not reconstruct
at all; and whatever was parked there on the day is treated as permanent.

### Watch positions

The **Watch position** tool (`O`) marks a place to watch from: where your ground camera stood, a
rooftop, a window across the road, a neighbouring car park. Give it a height (1.6 m is eye level)
and the survey shows what can be seen from there, on the plan and in 3-D, and lists which doors and
gates are in view. Survey photos taken on the ground are already plotted where they were taken, so
you can put a watch position on the same spot. It does not change the findings; it answers "what
would I see from here?".

### 2. Draw the plan

`survey new` builds the base map and opens the editor in your browser (a page served from
your own machine at `127.0.0.1`; nothing is uploaded). Trace what you see:

| Tool | Key | Draw |
|---|---|---|
| Boundary | `B` | The property line. Only ground inside it is assessed. |
| Building | `H` | Each building; set its height. |
| Wall, fence, hedge | `W` | Anything along a line. Pick solid wall, solid fence, see-through fence or hedge, and its height. |
| Trees, shrubs | `G` | Planting someone could stand in or behind. |
| Street, footpath | `S` | Where the public walks past. |
| Door, window, gate | `A` | Every way in. Snaps to the nearest wall or fence. |
| Camera | `C` | Existing cameras. Drag the handle to aim; set lens angle, range and height. |
| Watch position | `O` | A place to watch from, to see what it sees. |

Double-click or `Enter` finishes a shape, `Esc` cancels, `Delete` removes the selected item,
`Ctrl+Z` undoes. Drag corners to reshape; drag the hollow dots to add a corner. Every change is saved
to `site.json` and the findings on the right update as you draw, so you can try things: lower a
hedge, move a camera, add a recommended camera to the plan and see what is still unseen.

Re-run `survey new` on the same project folder to add photos; the plan is kept.
`survey edit my_site` reopens it later.

### People, vehicles and objects in the photos

Survey runs the same detectors as `isr triage` over the drone photos and puts every hit on the site
plan, so a person or a vehicle is reported by where they are on the site, not just by coordinates:
inside or outside the boundary, on camera or in ground nobody can see, how far from the nearest door
or gate, and whether it is against a fence. Something photographed several times is one hit.

In the editor each hit appears as a pink marker and a card with a close-up. **Confirm** it and say
what it is (person, vehicle, object) or **Dismiss** it. Expect false hits: the detector flags
anything whose colour stands out, including roof edges and garden furniture. Only confirmed hits
reach the report, under *Seen during the survey*, and a confirmed object left against a fence, or
anything confirmed in unobserved ground, becomes a finding.

```bash
survey new E:\DCIM -o my_site -c red        # also look for a known colour (clothing, a vehicle)
survey new E:\DCIM -o my_site --ai          # add the AI person/object detector
survey new E:\DCIM -o my_site --no-detect   # plan and report only
```

`-c`, `--sensitivity` and `--ai` behave as described under [Detectors](#detectors). To search open
country for a missing person, `isr triage` and `isr plan` remain the tools for that job.

### 3. Report

**Generate report** (or `survey report my_site --open`) writes into the project folder:

| File | Use |
|---|---|
| `report.html` | The assessment, written to hand to a client. No scripts, works offline; **Print / save PDF** in the page gives a PDF. |
| `site.geojson`, `site.kml` | Plan, cameras, access points and blind spots for Google Earth / QGIS |
| `assessment.csv` | One row per access point, blind spot, camera and perimeter opening |
| `assessment.json`, `site.json` | Everything, for your own tooling |

![Site report](docs/site_report.jpg)

Reports are headed **Ronin Sec** by default. Fill in the client, site address, your name and a job
reference under **Site** in the editor (and change "Prepared by" there if a job goes out under
another name). Notes typed against any item (a damaged fence panel, a gate that does
not latch) are printed beside it.

The report covers:

* **Priority findings**: a ranked list, each with what to do about it.
* **Sightlines**: the ground a passer-by on the street or an occupant at a window can see, and a
  what-if table of how much each hedge, fence or planting hides.
* **Blind spots**: every pocket nobody can see, with its size, what screens it, which doors it
  reaches, and its coordinates.
* **Camera placement**: recommended mounting points on building corners and walls with height, aim
  and what each one covers, chosen to deal with the worst-exposed access points first.
* **Guard posts and patrols** (when you set a number of guards under **Site**): the best places
  for an officer to stand, a walking patrol loop round everything those posts cannot see, and a
  recommended and an alternative deployment for each headcount from one officer up to your number,
  each with what it gains, what it leaves exposed, and what to do about it. Click an option in the
  editor to draw it on the plan.
* **Access points**: for each door, window, gate and driveway: on camera or not, seen from the
  street or not, how much hidden ground surrounds it, and the survey photos that show it.
* **Seen during the survey**: confirmed people, vehicles and objects, each with a close-up and where
  it was in relation to the plan.
* **Perimeter**: a map of the boundary with every opening marked, how much of it is closed, each gap
  not marked as a gate with its length and position, and a schedule of every wall, fence and hedge
  with type, height and length.

### How it works, and its limits

The plan is turned into a 0.5 m grid of obstruction heights. A spot counts as seen when a straight
line from the observer to a person standing there clears everything in between, taking height into
account: a camera at 2.7 m sees over a 1.2 m garden wall, a passer-by does not see over a 1.8 m hedge.

* **The tool measures the plan you draw; it does not recognise buildings or fences in the photos.**
  Results are as good as the tracing and the heights you enter.
* Positions come from consumer GPS and are good to a few metres. The ground is treated as flat.
* Camera coverage is geometric: an unobstructed view within the stated range and angle. It says
  nothing about image quality, night performance or whether anyone reviews the footage.
* The guard plan is geometry too: what a standing officer can see at eye height, and walking
  distance round buildings and through marked gates at 1.2 m/s with 20 s per stop. It knows nothing
  about shelter, radio cover, breaks, shift patterns or the client's actual threat. Walk a route
  before adopting it.
* Lighting, locks, glazing, alarms and upper floors are not assessed. They need a walk-through.

---

## Stakeout: a fixed camera's pattern of activity

Point a camera at a gate, a door or an approach and leave it for a day or a week. A trail camera
that fires on movement is ideal (it saves space); one that shoots every few minutes works too. Then
give ISRStudio the folder of photos: **New stakeout**, or drag the folder or card onto the Stakeout section.

It reads the time each photo was taken (from the camera's own stamp, or the file's date if there is
none), compares every frame with the same view at nearby times to find real movement, and groups
movement less than a minute apart into one event. You get:

* **When things move**: a week-by-hour grid of movements an hour, with the hours nobody watched hatched.
* **Out of hours**: every movement in the hours you say the site is closed, with the frame.
* **Where in the view**: a heat overlay on the camera's picture.
* **What it means for guarding**: the busiest windows (when a posted guard earns most), whether
  weekends differ, movement in the small hours to time patrols for, and the dead hours.

**Weather and light.** A camera left outside fires on plenty that is not a person. Give the
camera's location (latitude, longitude) when you start the stakeout and two things happen:

* the **sun's position** is worked out for every photo (no internet needed), so movements are
  split into daylight and darkness and broad changes at dawn and dusk are recognised;
* with *Fetch the weather* ticked, the hourly **weather history** for those days is fetched from
  [Open-Meteo](https://open-meteo.com), a free open weather service that needs no account. Only
  the location rounded to about a kilometre and the dates are sent.

A change in the picture is then **set aside** rather than counted when it looks like weather or
light: scattered patches instead of one shape; in a part of the view that changes far more on windy
hours than calm ones (a hedge, a flag); a broad broken-up change at dawn or dusk; or one tiny
change in a single frame during gusts or rain. Set-aside frames are shown at the end of the report
with the reason, so you can check them. In the demo week, two blustery days set the camera off on a
hedge about fifty times; with the weather history those are set aside, and without it most of them
are counted as movement.

**Linking it to a survey.** In the Survey editor's **Site** tab, attach a stakeout and say which
door or gate it watched. The survey then reports movement while closed as a finding, shows the
activity grid in the client report, and adds timing to the guard plan: which post sees that
entrance, when it is busiest, and when to time a patrol pass.

Limits: it counts changes in the picture. The weather and light checks reduce false movements but
do not remove them: headlights, a shadow crossing and animals still count, and a small distant
figure can be missed. Nobody is identified. One camera sees one view.
Photos only for now, not video. Only watch a site you are engaged to assess, with the owner's
agreement, and follow local rules on recording people.

---

## Other commands

```bash
isr info photo.JPG       # show what the tools read from a photo (GPS, altitude, gimbal, scale)
survey new --help
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
the `.kmz` structure, Art's ranking of degraded photos, duplicate grouping, horizon detection,
and that edits keep metadata and never touch the originals. The app tests run a job in each
section through the ISRStudio server. The site tests build the demo property
(`isr/site/demo.py`, which also writes a synthetic OpenDroneMap result) and check the map-projection
maths against known values, GeoTIFF import and measured heights, line-of-sight geometry, base-map placement to within 0.6 m (1 m from
angled shots alone), that
hidden doors rank highest and the recommended cameras cover them, the report, and the editor server.

## Safety and legal

These tools are for planning and for analysing imagery after the flight. You remain responsible for
airspace authorisation, staying under 120 m / 400 ft (the planner refuses anything higher unless
you pass `--allow-high`), keeping the aircraft in sight, and coordinating with whoever is running the
search. In the US, emergency operations outside Part 107 limits go through the FAA's SGI process,
coordinated with the responsible agency.
