# mantis

A collection of scripts and Blender projects for post-processing raw files captured from the SFI AutoShip's Lightweight Sensor Rig.

The rig carries two polarization cameras side by side. Each `.raw` capture holds one frame from each camera, stored as packed 12-bit data. The camera sensors combine a polarizer mosaic with a Bayer colour mosaic. `mantis` turns these captures into usable images (polarization intensities, Stokes vectors, colour images and AoLP/DoLP visualisations). It then loads the results into Blender's compositor for further processing and rendering.

> **Status:** Work in progress. A **stereoscopy** pipeline that uses both cameras together is under active development and is not part of this repository yet. Expect the scripts and folder layout to change.

---

## Repository layout

```
mantis/
├── parser/
│   ├── lucid_parser_new.py      # Recommended parser (PyTorch, GPU-accelerated)
│   ├── lucid_parser.py          # Original parser (NumPy/OpenCV), kept for reference
│   ├── weights_half_pol.pt      # Demosaicing kernel -> oriented intensities
│   ├── weights_half_stokes.pt   # Demosaicing kernel -> Stokes vector
│   ├── weights_full_pol.pt      # Full-resolution kernels (not used yet)
│   ├── weights_full_stokes.pt
│   ├── input/                   # Put your raw recordings here (git-ignored)
│   └── output/                  # Parser output ends up here (git-ignored)
└── compositing/
    ├── mantis_compositor.blend                       # General compositor setup
    ├── mantis_compositor_stokes_vector.blend         # Node tree for Stokes vector output
    ├── mantis_compositor_oriented_intensities.blend  # Node tree for oriented intensity output
    └── load_stokes_sequence_to_compositor.py         # Loads a parsed S0/S1/S2 sequence into the compositor
```

---

## Parser

The repository has two parser scripts:

| Script | Description |
| --- | --- |
| **`lucid_parser_new.py`** | ✅ **Highly recommended.** Demosaics with dedicated convolution kernels in PyTorch, which runs on the GPU when CUDA is available. It supports both cameras and every output type listed below. |
| `lucid_parser.py` | The original NumPy/OpenCV parser. It is kept for reference and because `lucid_parser_new.py` imports some of its helper functions (`_unpack_12p`, `_write_rgb`). Do not delete it, but use the new parser for actual work. |

### Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/get-started/locally/). Install a CUDA build for GPU acceleration; the parser falls back to the CPU otherwise.
- NumPy
- OpenCV (`opencv-python`)

```bash
cd parser
python -m venv venv
# Windows: venv\Scripts\activate   |   Linux/macOS: source venv/bin/activate
pip install torch numpy opencv-python
```

### Usage

Place your recordings under `parser/input/`, for example:

```
parser/input/2026-08-15/08-38-33_images/000001.raw
parser/input/2026-08-15/08-38-33_images/000002.raw
...
```

Then edit the `__main__` block at the bottom of `lucid_parser_new.py` to set the dates, times and flags you want, and run it **from inside the `parser/` folder**. The kernel weights are loaded with a relative path, so running it from anywhere else fails.

```bash
cd parser
python lucid_parser_new.py
```

You can also import it and call `convert_raw` directly:

```python
from lucid_parser_new import convert_raw

convert_raw(
    input_path="./input/2026-08-15/08-38-33_images/",  # a folder of .raw files, or a single .raw file
    output_path="./output/2026-08-15_08-38-33_images",
    output_name=None,          # None -> keeps the original frame number as the file name
    output_suffix="png",       # "png"/"tif" -> 16-bit output, "jpg" -> 8-bit output
    cam0=True,                 # left camera
    cam1=False,                # right camera
    oriented_intensities=False,
    stokes_vector=True,
    dolp_aolp=True,
    color=False,
    overwrite=False,           # False -> skip frames whose outputs already exist
)
```

### Output flags

| Flag | Output folder | Description |
| --- | --- | --- |
| `cam0` / `cam1` | `cam0/`, `cam1/` | Which camera(s) to process. Each `.raw` file holds one frame from each camera. |
| `oriented_intensities` | `camX/oriented_intensities/{i000,i045,i090,i135}/` | Demosaiced RGB intensity for each of the four polarizer angles. |
| `stokes_vector` | `camX/partial_stokes_vector/{s0,s1,s2}/` | The linear (partial) Stokes vector components S0, S1 and S2 in RGB. |
| `dolp_aolp` | `camX/aolp/`, `camX/dolp/`, `camX/dolp_aolp/` | Angle and degree of linear polarization, plus a combined colour-coded visualisation. |
| `color` | `camX/color/` | An exposure-normalised colour image computed from S0. |
| `overwrite` | – | If `False`, a raw frame is skipped when every output file the enabled flags would produce already exists. Enabling an extra flag later re-processes only the frames that are missing that output. |

### Recommended settings for simple parsing

If you do not need both cameras (that is, you are not doing stereo work), the recommended setup is:

```python
cam0=True,            # only one of the cameras
cam1=False,
stokes_vector=True,
dolp_aolp=True,
oriented_intensities=False,
color=False,
```

- **`stokes_vector=True`**: The Stokes vector holds all the linear polarization information. You can derive intensity, AoLP and DoLP from it.
- **Only one camera flag set to `True`**: This halves the processing time and the disk usage.
- **`dolp_aolp=True`, even though you can compute it from the Stokes vector yourself**: The parser's AoLP/DoLP uses a more specialised demosaicing kernel to compute the intensity values than most users would write themselves. It also writes a ready-to-view colour visualisation.

---

## Compositing

The `compositing/` folder contains Blender projects with compositor node trees, one for each output type the parser produces:

| Blend file | Use with parser output |
| --- | --- |
| `mantis_compositor_stokes_vector.blend` | `partial_stokes_vector/` (`stokes_vector=True`) |
| `mantis_compositor_oriented_intensities.blend` | `oriented_intensities/` (`oriented_intensities=True`) |
| `mantis_compositor.blend` | General compositor setup |

The helper script runs from Blender's **Scripting** workspace:

- **`load_stokes_sequence_to_compositor.py`** loads the S0/S1/S2 image sequences from `parser/output/<recording>/<cam>/partial_stokes_vector/` into the compositor nodes `S0 - seq`, `S1 - seq` and `S2 - seq`, and sets the scene frame range to match. Set `IMAGE_FOLDER`, `CAM_FODLER` and `SUFFIX` at the top of the script before you run it.
The script resolves paths relative to the `.blend` file, so save the file inside `compositing/` before you run it. They support Blender 4.x and 5.x.

---

## Roadmap

- [ ] Stereoscopy / depth reconstruction using both cameras
- [ ] Full-resolution demosaicing (`weights_full_*.pt`)
- [ ] Command-line interface for the parser instead of editing `__main__`

## Contributing

This is an early-stage research project. Issues and suggestions are welcome, but expect breaking changes while the stereoscopy work is in progress.
