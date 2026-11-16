import os
import glob

import numpy as np
import torch
from torch.nn.functional import conv2d

from lucid_parser import _unpack_12p, _write_rgb


# The sensor carries a polarizer mosaic on top of a Bayer mosaic (the layout of
# Sony's Polarsens parts, e.g. IMX250MYR; measured from the raw data rather than
# taken from a datasheet).
# Every 2x2 block of pixels holds the four polarizer orientations, and every 2x2
# block of *those* blocks holds one RGGB Bayer quad, i.e. a 4x4 repeating unit:
#
#     [R090, R045, G090, G045]
#     [R135, R000, G135, G000]
#     [G090, G045, B090, B045]
#     [G135, G000, B135, B000]
#
# So de-interleaving with the offsets below leaves four half-resolution RGGB
# Bayer images, one per orientation.


# CONSTANTS
device = "cuda" if torch.cuda.is_available() else "cpu"

BT709 = torch.tensor(
    [
        [0.2126, 0.7152, 0.0722],
        [-0.1146, -0.3854, 0.5],
        [0.5, -0.4542, -0.0458],
    ],
    dtype=torch.float32,
    device=device,
)

RGB2YUV = BT709

MUNSELL_MAP_RGB = torch.tensor(
    [
        [201, 130, 134],
        [201, 130, 127],
        [201, 131, 118],
        [200, 133, 109],
        [197, 135, 100],
        [193, 137, 94],
        [187, 140, 86],
        [181, 143, 79],
        [173, 146, 75],
        [167, 149, 72],
        [160, 151, 73],
        [151, 154, 78],
        [141, 156, 85],
        [127, 159, 98],
        [115, 160, 110],
        [101, 162, 124],
        [92, 163, 134],
        [87, 163, 141],
        [82, 163, 148],
        [78, 163, 154],
        [73, 163, 162],
        [70, 162, 170],
        [70, 161, 177],
        [73, 160, 184],
        [82, 158, 189],
        [93, 156, 193],
        [104, 154, 195],
        [117, 151, 197],
        [128, 149, 198],
        [141, 145, 198],
        [152, 142, 196],
        [160, 140, 193],
        [168, 138, 189],
        [177, 135, 182],
        [183, 134, 176],
        [188, 132, 169],
        [193, 131, 160],
        [196, 130, 153],
        [198, 130, 146],
        [200, 130, 140],
        [201, 130, 134],
    ],
    dtype=torch.float32,
    device=device,
)

# WEIGHTS_FULL_POL = torch.load("weights_full_pol.pt")*(1/2**12)      # Convolution kernel weigths for demosaeicing unpacked raw files from 
# WEIGHTS_FULL_STO = torch.load("weights_full_stokes.pt")*(1/2**12)   # The input is 12 bit, so we simply normalize, we could do this anywhere in the beginning however it is easiest to do simply in the conv kernel
WEIGHTS_HALF_POL = torch.load("weights_half_pol.pt")*(1/2**12)
WEIGHTS_HALF_STO = torch.load("weights_half_stokes.pt")*(1/2**12)


# HELPER FUNCTIONS (CORE)
@torch.jit.script
def colormap(cmap: torch.Tensor, args: torch.Tensor) -> torch.Tensor:
    "args: float in [0, 1]"
    "cmap: n x 3"

    n = cmap.size(0)
    idx0 = (args * n).int() % n
    v0 = cmap[idx0]
    v1 = cmap[(idx0 + 1) % n]
    reminder = (args * n - idx0).unsqueeze(-1)
    return v0 * (1 - reminder) + v1 * reminder

@torch.jit.script
def rgb_to_y(rgb: torch.Tensor, mat: torch.Tensor = BT709) -> torch.Tensor:
    return torch.einsum("j,...j->...", mat[0], rgb)

@torch.jit.script
def dolp_aolp_to_rgb(
    dolp01: torch.Tensor, aolp01: torch.Tensor, cmap: torch.Tensor = MUNSELL_MAP_RGB
) -> torch.Tensor:
    return colormap(cmap, aolp01) * dolp01.unsqueeze(-1)

@torch.jit.script
def con_to_chanlast(data: torch.Tensor) -> torch.Tensor:
    assert data.is_contiguous()
    while data.ndim < 4:
        data = data.unsqueeze(0)
    return data.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)

@torch.jit.script
def chanlast_to_con(data: torch.Tensor) -> torch.Tensor:
    assert data.is_contiguous(memory_format=torch.channels_last)
    return data.permute(0, 2, 3, 1).to(memory_format=torch.contiguous_format)

@torch.jit.script
def from_tile2d(data: torch.Tensor, shape_start: int) -> torch.Tensor:
    ss = shape_start
    return data.swapaxes(ss + 1, ss + 2).flatten(ss, ss + 1).flatten(ss + 1, ss + 2)

def torch_image_to_numpy_image(img: torch.Tensor, scale: float = 1.0) -> np.ndarray:
    """
    Tensor (any device/dtype), last dim = channels if present -> uint16 numpy array.
    Floats are multiplied by `scale` (e.g. 65535 for 0..1 data), rounded and clipped.
    """
    img = img.detach()
    if img.is_floating_point():
        img = (img * scale).round()
    img = img.clamp(0, 65535).to(torch.int32)    # int32: uint16 has patchy torch support
    return img.cpu().numpy().astype(np.uint16)

def _restack_unpacked(img: torch.Tensor) -> torch.Tensor:
    """(H, W) mosaic -> (H/4, W/4, 16), channel = x0 + 2*y0 + 4*x1 + 8*y1."""
    H, W = img.shape
    assert H % 4 == 0 and W % 4 == 0
    return (
        img.to(torch.float32)                     # kernel outputs float; uint16 has limited torch op support
        .reshape(H // 4, 2, 2, W // 4, 2, 2)      # Y, y1, y0, X, x1, x0
        .permute(0, 3, 1, 4, 2, 5)                # Y, X, y1, x1, y0, x0
        .reshape(H // 4, W // 4, 16)
    )

def _expected_outputs(
    output_path: str,
    out_name: str,
    jpg_out_name: str,
    cam0: bool,
    cam1: bool,
    oriented_intensities: bool,
    stokes_vector: bool,
    dolp_aolp: bool,
    color: bool,
) -> list[str]:
    """Every file convert_raw writes for one raw frame with the given flags.

    Keep this in sync with the _write_rgb calls in convert_raw.
    """
    cams = [cam for cam, enabled in (("cam0", cam0), ("cam1", cam1)) if enabled]
    paths = []
    for cam in cams:
        cam_dir = os.path.join(output_path, cam)
        if oriented_intensities:
            for angle in ("i090", "i045", "i135", "i000"):
                paths.append(os.path.join(cam_dir, "oriented_intensities", angle, out_name))
        if stokes_vector:
            for component in ("s0", "s1", "s2"):
                paths.append(os.path.join(cam_dir, "partial_stokes_vector", component, out_name))
        if color:
            paths.append(os.path.join(cam_dir, "color", out_name))
        if dolp_aolp:
            paths.append(os.path.join(cam_dir, "aolp", out_name))
            paths.append(os.path.join(cam_dir, "dolp", out_name))
            paths.append(os.path.join(cam_dir, "dolp_aolp", jpg_out_name))
    return paths

# SHELL SCRIPT
def convert_raw(
    input_path: str, 
    output_path: str, 
    output_name: str | None = None,
    output_suffix: str = "png", 
    cam0: bool = True,
    cam1: bool = True,
    oriented_intensities: bool = False,
    stokes_vector: bool = True,
    dolp_aolp: bool = False,
    color: bool = False,
    overwrite: bool = False
):
    if not (cam0 or cam1):
        return

    RES = (2448, 2048)
    res_x, res_y = RES
    frame_bytes = (res_x * res_y * 3) // 2 # 12-bit depth
    suffix = output_suffix.lstrip(".")

    # Collect the raw files to process.
    # input_path may be either a single .raw file or a folder containing .raw files.
    # Each raw file contains two images of equal size (2448 x 2048),
    # which are unravelled into cam0 (left) and cam1 (right).
    if os.path.isfile(input_path):
        if not input_path.lower().endswith(".raw"):
            raise ValueError(f"Input file is not a .raw file: {input_path}")
        raw_paths = [input_path]
    elif os.path.isdir(input_path):
        raw_paths = sorted(glob.glob(os.path.join(input_path, "*.raw")))
        if not raw_paths:
            raise FileNotFoundError(f"No .raw files found in {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


    jpg_out_name = None
    value_range = 65535 if suffix in ("png", "tif", "tiff") else 255

    for raw_path in raw_paths:
        
        if output_name == None:
            stem = os.path.splitext(os.path.basename(raw_path))[0]
            # drop the first two characters of the original file name if both are
            # the character '0', otherwise throw an error
            if stem[:2] != "00":
                raise ValueError(
                    f"Expected the file name {stem!r} to start with '00', "
                    f"refusing to truncate it"
                )
            out_name = f"{stem[2:]}.{suffix}"
            jpg_out_name = f"{stem[2:]}.jpg"
        else:
            out_name = f"{output_name}.{suffix}"
            jpg_out_name = f"{output_name}.jpg"

        if not overwrite:
            expected = _expected_outputs(
                output_path, out_name, jpg_out_name,
                cam0=cam0, cam1=cam1,
                oriented_intensities=oriented_intensities,
                stokes_vector=stokes_vector,
                dolp_aolp=dolp_aolp,
                color=color,
            )
            if expected and all(os.path.exists(path) for path in expected):
                print(f"Skipping {raw_path}: all {len(expected)} output file(s) already exist")
                continue

        buffer = np.fromfile(raw_path, dtype=np.uint8)
        if buffer.size != 2 * frame_bytes:
            raise ValueError(
                f"{raw_path}: expected {2 * frame_bytes} bytes for two "
                f"{res_x}x{res_y} BayerRG12p frames, got {buffer.size}"
            )

        if cam0:
            cam0_buffer = _unpack_12p(buffer[:frame_bytes], res_y, res_x)
            cam0_buffer = torch.from_numpy(cam0_buffer).to(device)
            cam0_buffer = _restack_unpacked(cam0_buffer)
        if cam1:
            cam1_buffer = _unpack_12p(buffer[frame_bytes:], res_y, res_x)
            cam1_buffer = torch.from_numpy(cam1_buffer).to(device)
            cam1_buffer = _restack_unpacked(cam1_buffer)
        
        # We work with the data on the contiguous data-order format (NCHW), while the convolution wants the NHWC data format: NCHW -> NHWC, conv2d, NHWC -> NCHW.

        if oriented_intensities:
            if cam0:
                cam0_oriented_intensities   : torch.Tensor
                cam0_oriented_intensities   = chanlast_to_con(conv2d(con_to_chanlast(cam0_buffer), WEIGHTS_HALF_POL, padding=0))   # torch.Size([1, 508, 608, 48])
                cam0_oriented_intensities   = cam0_oriented_intensities.unflatten(-1, (2, 2, 3, 4))                         # torch.Size([1, 508, 608, 2, 2, 3, 4])
                cam0_oriented_intensities   = from_tile2d(cam0_oriented_intensities, 1)                                     # torch.Size([1, 1016, 1216, 3, 4])
                cam0_imgs = torch_image_to_numpy_image(cam0_oriented_intensities[0], scale=value_range)   # (H, W, 3, 4) uint16, one transfer # 
            if cam1:
                cam1_oriented_intensities   : torch.Tensor
                cam1_oriented_intensities   = chanlast_to_con(conv2d(con_to_chanlast(cam1_buffer), WEIGHTS_HALF_POL, padding=0))   # torch.Size([1, 508, 608, 48])
                cam1_oriented_intensities   = cam1_oriented_intensities.unflatten(-1, (2, 2, 3, 4))                         # torch.Size([1, 508, 608, 2, 2, 3, 4])
                cam1_oriented_intensities   = from_tile2d(cam1_oriented_intensities, 1)                                     # torch.Size([1, 1016, 1216, 3, 4])
                cam1_imgs = torch_image_to_numpy_image(cam1_oriented_intensities[0], scale=value_range)   # (H, W, 3, 4) uint16, one transfer 

            angles = ("i090", "i045", "i135", "i000")   # check this matches the order of your last dimension
            for i, angle in enumerate(angles):
                if cam0:
                    _write_rgb(os.path.join(output_path, "cam0", "oriented_intensities", angle), out_name, cam0_imgs[..., i], value_range=value_range)
                if cam1:
                    _write_rgb(os.path.join(output_path, "cam1", "oriented_intensities", angle), out_name, cam1_imgs[..., i], value_range=value_range)

        if stokes_vector or dolp_aolp or color:
            if cam0:
                cam0_partial_stokes : torch.Tensor
                cam0_partial_stokes = chanlast_to_con(conv2d(con_to_chanlast(cam0_buffer), WEIGHTS_HALF_STO, padding=0))
                cam0_partial_stokes = cam0_partial_stokes.unflatten(-1, (2, 2, 3, 3))
                cam0_partial_stokes = from_tile2d(cam0_partial_stokes, 1)
                cam0_imgs = torch_image_to_numpy_image(cam0_partial_stokes[0], scale=value_range) # [0] select only the first batch -> N batches (in our case always 1, therefor we select the first and only element by indexing the 0th)
            if cam1:
                cam1_partial_stokes : torch.Tensor
                cam1_partial_stokes = chanlast_to_con(conv2d(con_to_chanlast(cam1_buffer), WEIGHTS_HALF_STO, padding=0))
                cam1_partial_stokes = cam1_partial_stokes.unflatten(-1, (2, 2, 3, 3))
                cam1_partial_stokes = from_tile2d(cam1_partial_stokes, 1)
                cam1_imgs = torch_image_to_numpy_image(cam1_partial_stokes[0], scale=value_range)

            stokes_components = ("s0", "s1", "s2")   # check this matches the order of your last dimension
            if stokes_vector:
                for i, component in enumerate(stokes_components):
                    if cam0:
                        _write_rgb(os.path.join(output_path, "cam0", "partial_stokes_vector", component), out_name, cam0_imgs[..., i], value_range=value_range)
                    if cam1:
                        _write_rgb(os.path.join(output_path, "cam1", "partial_stokes_vector", component), out_name, cam1_imgs[..., i], value_range=value_range)

            if color:
                if cam0:
                    cam0_s0 = torch.clamp(cam0_partial_stokes[..., 0], 1e-6, 1)
                    cam0_gain = 1.0 / torch.quantile(cam0_s0, 0.95)
                    cam0_s0 *= cam0_gain
                    _write_rgb(os.path.join(output_path, "cam0", "color"), out_name, torch_image_to_numpy_image(cam0_s0[0], scale=value_range))
                if cam1:
                    cam1_s0 = torch.clamp(cam1_partial_stokes[..., 0], 1e-6, 1)
                    cam1_gain = 1.0 / torch.quantile(cam1_s0, 0.95)
                    cam1_s0 *= cam1_gain
                    _write_rgb(os.path.join(output_path, "cam1", "color"), out_name, torch_image_to_numpy_image(cam1_s0[0], scale=value_range))

            if dolp_aolp:
                if cam0:
                    cam0_s0y = rgb_to_y(cam0_partial_stokes[..., 0])
                    cam0_s1y = rgb_to_y(cam0_partial_stokes[..., 1])
                    cam0_s2y = rgb_to_y(cam0_partial_stokes[..., 2])
                    cam0_dolp = torch.clamp(torch.sqrt(cam0_s1y**2 + cam0_s2y**2) / cam0_s0y, 0, 1) # This is correct
                    cam0_aolp = ((torch.arctan2(cam0_s2y, cam0_s1y) / (torch.pi)) + 1) / 2
                    cam0_dolp_aolp = dolp_aolp_to_rgb(cam0_dolp, cam0_aolp) # *(1/2**8) # Normalize the output [0, 255] -> [0, 1]
                    _write_rgb(os.path.join(output_path, "cam0", "aolp"), f"{out_name}", torch_image_to_numpy_image(cam0_aolp[0], scale=value_range), value_range=value_range)
                    _write_rgb(os.path.join(output_path, "cam0", "dolp"), f"{out_name}", torch_image_to_numpy_image(cam0_dolp[0], scale=value_range), value_range=value_range)
                    _write_rgb(os.path.join(output_path, "cam0", "dolp_aolp"), jpg_out_name, torch_image_to_numpy_image(cam0_dolp_aolp[0], scale=1.0), value_range=255)
                if cam1:
                    cam1_s0y = rgb_to_y(cam1_partial_stokes[..., 0])
                    cam1_s1y = rgb_to_y(cam1_partial_stokes[..., 1])
                    cam1_s2y = rgb_to_y(cam1_partial_stokes[..., 2])
                    cam1_dolp = torch.clamp(torch.sqrt(cam1_s1y**2 + cam1_s2y**2) / cam1_s0y, 0, 1)
                    cam1_aolp = (torch.arctan2(cam1_s2y, cam1_s1y) / (torch.pi) + 1) / 2 
                    cam1_dolp_aolp = dolp_aolp_to_rgb(cam1_dolp, cam1_aolp) # *(1/2**8)
                    _write_rgb(os.path.join(output_path, "cam1", "aolp"), f"{out_name}", torch_image_to_numpy_image(cam1_aolp[0], scale=value_range), value_range=value_range)
                    _write_rgb(os.path.join(output_path, "cam1", "dolp"), f"{out_name}", torch_image_to_numpy_image(cam1_dolp[0], scale=value_range), value_range=value_range)
                    _write_rgb(os.path.join(output_path, "cam1", "dolp_aolp"), jpg_out_name, torch_image_to_numpy_image(cam1_dolp_aolp[0], scale=1.0), value_range=255)
                
        if len(raw_paths) > 1:
            print(f"{raw_path} -> {out_name}")

        
if __name__ == "__main__":
    dates = ("2026-08-18",)
    times = ("16-35-48_images", )

    for date in dates:
        for time in times:
            convert_raw(
                input_path=f"./input/{date}/{time}/",
                output_path=f"./output/2026-00-18_{time}",
                output_name=None,
                output_suffix="jpg",
                cam0=True,
                cam1=True,
                oriented_intensities=False,
                stokes_vector=True,
                dolp_aolp=True,
                color=False,
                overwrite=True
            )

    # dates = ("2026-09-09-003",)
    # times = ("11-26-09_images",)

    # for date in dates:
    #     for time in times:
    #         convert_raw(
    #             input_path=f"./input/{date}/{time}/",
    #             output_path=f"./output/2026-09-09_{time}",
    #             output_name=None,
    #             output_suffix="jpg",
    #             cam0=True,
    #             cam1=True,
    #             oriented_intensities=False,
    #             stokes_vector=True,
    #             dolp_aolp=True,
    #             color=False,
    #             overwrite=True
    #         )