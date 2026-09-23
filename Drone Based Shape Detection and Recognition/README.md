# Shape Detector

Detect and classify geometric cardboard cutouts in static images. The main
script reads one image or a folder of images, finds supported shapes, writes
annotated copies, and appends detection metadata to a CSV file.

## Supported Shapes

- Triangle
- Square
- Rectangle
- Pentagon
- Hexagon
- Heptagon
- Octagon
- Nonagon
- Decagon
- Circle
- Ellipse

## Project Layout

- `shape_detector.py` - main detection and annotation CLI.
- `generate_test_shapes.py` - creates simple synthetic shape images in `input/`.
- `generate_composite_test_images.py` - overlays supported shapes onto background photos and writes ground-truth JSON files.
- `generate_print_templates.py` - creates A3 printable PDF templates in `templates/`.
- `rename_images.py` - renames images in `test/` to `test1`, `test2`, etc.
- `input/` - default detector input folder.
- `output/` - default folder for annotated images and `results.csv`.
- `backgrounds/` - source photos for composite test image generation.
- `composite_input/` - generated composite test images and ground-truth JSON files.
- `templates/` - generated printable shape templates.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv shape_env
.\shape_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Detect Shapes

Run the detector against the default `input/` folder:

```powershell
python shape_detector.py
```

Run it against a specific image or folder:

```powershell
python shape_detector.py input --output output
python shape_detector.py test\test1.jpg --output output
```

The detector writes annotated images to the output folder and appends detections
to `output/results.csv`. The CSV includes filename, detected shape, vertex count,
area, bounding box, circularity, and confidence.

## Detection Options

The default method is `auto`, which compares adaptive thresholding and Canny edge
detection and chooses the cleaner contour set.

```powershell
python shape_detector.py input --method auto
python shape_detector.py input --method adaptive
python shape_detector.py input --method canny
python shape_detector.py input --method hsv --hsv-target cardboard
python shape_detector.py input --method hsv --hsv-target white
```

Useful tuning flags:

```powershell
python shape_detector.py input --min-area 2500
python shape_detector.py input --epsilon 0.025
python shape_detector.py input --shape-confidence 0.70
python shape_detector.py input --debug-mask debug_masks --verbose
```

For HSV mode, custom bounds can be provided as comma-separated `H,S,V` values:

```powershell
python shape_detector.py input --method hsv --hsv-lower 5,45,45 --hsv-upper 35,255,255
```

## Generate Test Assets

Create one simple test image per supported shape:

```powershell
python generate_test_shapes.py
```

Generate realistic composite images from photos in `backgrounds/`:

```powershell
python generate_composite_test_images.py backgrounds --output-dir composite_input --seed 7
```

Generate more than one composite per background:

```powershell
python generate_composite_test_images.py backgrounds --count-per-background 3
```

Generate A3 printable shape templates:

```powershell
python generate_print_templates.py
```

Rename images in `test/` sequentially:

```powershell
python rename_images.py
```

## Notes

- Supported input image extensions are `.bmp`, `.jpeg`, `.jpg`, `.png`, `.tif`,
  `.tiff`, and `.webp`.
- `opencv-python-headless` is used because the project processes files without a
  GUI window.
- Existing output files are not overwritten; duplicate annotated image names get
  `_2`, `_3`, and later suffixes.
