import glob
import os

import cv2
import numpy as np

from typing import Literal

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
POLARIZATION_OFFSETS = {
    "i000": (1, 1),
    "i045": (0, 1),
    "i090": (0, 0),
    "i135": (1, 0),
}

MAX_12BIT = 4095
# 12 bit -> 16 bit container. Exactly invertible: divide by 16 to get the raw ADU.
U16_PER_ADU = 16

# Rec.601 luma weights, applied to (..., 3) RGB arrays.
LUMA_RGB = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _unpack_12p(packed: np.ndarray, height: int, width: int) -> np.ndarray:
    """
    Unpack a LUCID/GenICam ``*12p`` buffer (BayerRG12p) into 12 bit samples.

    Three bytes hold two pixels, least significant bits first:

        byte0 = p0[7:0]
        byte1 = p1[3:0] << 4 | p0[11:8]
        byte2 = p1[11:4]

    Note this is *not* the GigE Vision ``12Packed`` layout, which orders the
    nibbles the other way around.
    """
    if packed.size != (height * width * 3) // 2:
        raise ValueError(
            f"Expected {(height * width * 3) // 2} packed bytes for a "
            f"{width}x{height} 12 bit frame, got {packed.size}"
        )

    triplets = packed.reshape(-1, 3).astype(np.uint16)
    pixels = np.empty(height * width, dtype=np.uint16)
    pixels[0::2] = triplets[:, 0] | ((triplets[:, 1] & 0x0F) << 8)
    pixels[1::2] = (triplets[:, 1] >> 4) | (triplets[:, 2] << 4)
    return pixels.reshape(height, width)


def _split_orientations(frame: np.ndarray) -> dict:
    """
    De-interleave one raw frame into four debayered RGB images, one per
    polarizer orientation. Returns float32 arrays still on the 0..4095 ADU
    scale, shaped (height/2, width/2, 3).
    """
    oriented = {}
    for orientation, (row, col) in POLARIZATION_OFFSETS.items():
        mosaic = np.ascontiguousarray(frame[row::2, col::2])
        # OpenCV names a Bayer pattern after the 2nd row / 2nd and 3rd columns,
        # so an RGGB mosaic is decoded with the "BayerBG" constants.
        oriented[orientation] = cv2.cvtColor(mosaic, cv2.COLOR_BayerBG2RGB).astype(np.float32)
    return oriented


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Luma of an RGB array. Works for the signed S1/S2 planes as well."""
    return rgb @ LUMA_RGB

# def _write_rgb(directory: str, filename: str, image: np.ndarray) -> None:
#     """
#     Write an RGB (or single channel) float array as a 16 bit image. Values are
#     expected to already be on the 0..65535 scale; they are clipped and rounded.
#     """
#     os.makedirs(directory, exist_ok=True)
#     path = os.path.join(directory, filename)

#     image = np.clip(image, 0, 65535).astype(np.uint16)
#     if image.ndim == 3:
#         image = image[..., ::-1]  # RGB -> BGR for OpenCV
#     if not cv2.imwrite(path, np.ascontiguousarray(image)):
#         raise IOError(f"cv2.imwrite failed for {path}")


ValueRange = Literal[1, 255, 65535]


def _write_rgb(
    directory: str,
    filename: str,
    image: np.ndarray,
    value_range: ValueRange = 65535,
    jpeg_quality: int = 95,
) -> None:
    """
    Write an RGB (or single channel) array to disk.

    value_range describes the scale the input values are on:
      65535 -> written as 16 bit PNG (filename must end in .png)
      255   -> written as 8 bit PNG or JPEG
      1     -> scaled to 0..255, written as 8 bit PNG or JPEG

    Values are clipped to the range and rounded to the nearest integer.
    The output format is taken from the filename extension.
    """
    if value_range not in (1, 255, 65535):
        raise ValueError(f"value_range must be 1, 255 or 65535, got {value_range}")

    ext = os.path.splitext(filename)[1].lower()
    image = np.asarray(image, dtype=np.float64)

    if value_range == 65535:
        if ext != ".png":
            raise ValueError("value_range=65535 is written as 16 bit PNG; use a .png filename")
        out = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        if ext not in (".png", ".jpg", ".jpeg"):
            raise ValueError(f"value_range={value_range} supports .png or .jpg/.jpeg, got '{ext}'")
        scaled = image * (255.0 / value_range)
        out = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
        if ext == ".png":
            params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        else:
            params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    if out.ndim == 3 and out.shape[2] == 3:
        out = out[..., ::-1]  # RGB -> BGR for OpenCV

    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)
    if not cv2.imwrite(path, np.ascontiguousarray(out), params):
        raise IOError(f"cv2.imwrite failed for {path}")


def convert_raw(
    input_path: str, 
    output_path: str, 
    output_suffix: str = "png"
):
    """
    Processes the raw data and saves them as images. This includes as allways the color channels,
    however in addition the polarization data of the light is saved if wanted. This does however leave you with options
        Select oriented_intensities if you want the rgb images at the four different orientation angles: 0deg, 45deg, 90deg, 135deg/-45deg
            - this option is one-to-one with the captured data and gives you the most versatility to post-analyze the images yourself
        Select aolp_dolp to get the angle and degree of linear polarization as a single image
        Select partial_stokes_vector to get the grayscale S0, S1 and S2 planes as a single image

    Every raw file holds two 2448x2048 BayerRG12p frames back to back, cam0 (left)
    first and cam1 (right) second, matching the order of the sidecar .json.

    Everything is written as 16 bit images, so use a suffix that supports that
    (png or tif; jpg will silently be reduced to 8 bit). The output layout is

        {output_path}/{cam}/oriented_intensities/{i000,i045,i090,i135}/
    """

    RES = (2448, 2048)
    res_x, res_y = RES

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

    frame_bytes = (res_x * res_y * 3) // 2

    for raw_path in raw_paths:
        stem = os.path.splitext(os.path.basename(raw_path))[0]
        # drop the first two characters of the original file name if both are
        # the character '0', otherwise throw an error
        if stem[:2] != "00":
            raise ValueError(
                f"Expected the file name {stem!r} to start with '00', "
                f"refusing to truncate it"
            )
        out_name = f"{stem[2:]}.{suffix}"

        if (
            os.path.exists(os.path.join(output_path, f"cam0/oriented_intensities/i000/{out_name}")) # or
            # os.path.exists(os.path.join(output_path, f"cam0/oriented_intensities/{out_name}"))
        ):
            print(f"Skipping {out_name} as it allready exists")
            continue

        buffer = np.fromfile(raw_path, dtype=np.uint8)
        if buffer.size != 2 * frame_bytes:
            raise ValueError(
                f"{raw_path}: expected {2 * frame_bytes} bytes for two "
                f"{res_x}x{res_y} BayerRG12p frames, got {buffer.size}"
            )

        cam0 = _unpack_12p(buffer[:frame_bytes], res_y, res_x)
        cam1 = _unpack_12p(buffer[frame_bytes:], res_y, res_x)

        for cam_name, cam in (("cam0", cam0), ("cam1", cam1)):
            cam_out = os.path.join(output_path, cam_name)

            # Split the frame into its i000, i045, i090, i135 channels
            # [i090, i045, i090, i045, ....]
            # [i135, i000, i135, i000, ....]
            # [i090, i045, i090, i045, ....]
            # [i135, i000, i135, i000, ....]
            #
            # and debayer each of the four (rggb) images
            # (res_y/2, res_x/2, channels[1]=rggb) -> (res_y/2, res_x/2, channels[3]=rgb)
            oriented = _split_orientations(cam)
            # i000, i045, i090, i135 = (
            #     oriented["i000"], oriented["i045"], oriented["i090"], oriented["i135"]
            # )
            for orientation, image in oriented.items():
                _write_rgb(
                    os.path.join(cam_out, "oriented_intensities", orientation),
                    out_name,
                    image * U16_PER_ADU,
                )

        print(f"{raw_path} -> {out_name}")


if __name__ == "__main__":
    # Recomended output: oriented_intensities
    # 10-17-29_images
    convert_raw(
        input_path='./input/2026-08-15/08-38-33_images',
        output_path='./output/2026-08-15_08-38-33_images'
    )