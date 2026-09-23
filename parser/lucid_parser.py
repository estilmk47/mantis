import glob
import os

import cv2
import numpy as np


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


def _write_rgb(directory: str, filename: str, image: np.ndarray) -> None:
    """
    Write an RGB (or single channel) float array as a 16 bit image. Values are
    expected to already be on the 0..65535 scale; they are clipped and rounded.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, filename)

    image = np.clip(image, 0, 65535).astype(np.uint16)
    if image.ndim == 3:
        image = image[..., ::-1]  # RGB -> BGR for OpenCV
    if not cv2.imwrite(path, np.ascontiguousarray(image)):
        raise IOError(f"cv2.imwrite failed for {path}")


def convert_raw(
    input_path: str, 
    output_path: str, 
    output_suffix: str = "png", 

    oriented_intesities: bool = False,

    color_formating: str = "s1", # Choose between 'max_pool', 'average' and 's1'
    color: bool = True,

    aolp_encoding: str = "sine_and_cosine", # Choose between 'sine_and_cosine' and 'normalized'
    aolp_dolp: bool = True, 

    partial_stokes_vector: bool = False
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
        {output_path}/{cam}/color/
        {output_path}/{cam}/partial_stokes_vector/
        {output_path}/{cam}/aolp_dolp/

    and the encodings, all of them exactly invertible, are

        oriented intensities, color   value  = adu * 16,            adu in [0, 4095]
        stokes S0                     value  = S0 * 8,              S0  in [0, 8190]
        stokes S1, S2                 value  = (S + 4096) * 8,      S   in [-4095, 4095]
        dolp                          value  = dolp * 65535,        dolp in [0, 1]
        aolp sine/cosine              value  = (t + 1) / 2 * 65535, t   in [-1, 1]
        aolp normalized               value  = (aolp / pi + 0.5) * 65535

    No white balance is applied; the raw ADUs are kept as captured. The per frame
    balance_ratio from the sidecar .json can be applied afterwards if wanted.
    """

    RES = (2448, 2048)
    res_x, res_y = RES

    suffix = output_suffix.lstrip(".")

    if color_formating not in ("s1", "max_pool", "average"):
        raise ValueError(f"Unknown color_formating {color_formating!r}")
    # The default value spells it "normalized", the branch below spelled it
    # "normalize"; accept both.
    if aolp_encoding not in ("sine_and_cosine", "normalized", "normalize"):
        raise ValueError(f"Unknown aolp_encoding {aolp_encoding!r}")

    # Loop through all the raw files in the input folder
    # The raw files contains two images of equal size of 2448 x 2048 resolution
    # Unravel these binaries into two variables cam0 (left) and cam1 (right)
    raw_paths = sorted(glob.glob(os.path.join(input_path, "*.raw")))
    if not raw_paths:
        raise FileNotFoundError(f"No .raw files found in {input_path}")

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
            i000, i045, i090, i135 = (
                oriented["i000"], oriented["i045"], oriented["i090"], oriented["i135"]
            )

            if oriented_intesities:
                for orientation, image in oriented.items():
                    _write_rgb(
                        os.path.join(cam_out, "oriented_intensities", orientation),
                        out_name,
                        image * U16_PER_ADU,
                    )

            if not (color or partial_stokes_vector or aolp_dolp):
                continue

            # Get partial stokes vector
            S0 = i000 + i090
            S1 = i000 - i090
            S2 = i045 - i135

            if color:
                if color_formating == "s1":
                    # S0 is the total intensity of the i000/i090 pair, so half of
                    # it is on the same scale as a single orientation. TODO: consider not halfing it as the polarization filters do block light as well
                    color_image = S0 / 2
                elif color_formating == "max_pool":
                    # max select the maximum pixel value for each color channel selected from the four orientations i000, i045, i090, i135
                    color_image = np.maximum(np.maximum(i000, i045), np.maximum(i090, i135))
                elif color_formating == "average":
                    # average the pixel value for each coolor channel selected from the four orientations i000, i045, i090, i135
                    color_image = (i000 + i045 + i090 + i135) / 4

                _write_rgb(os.path.join(cam_out, "color"), out_name, color_image * U16_PER_ADU)

            if not (aolp_dolp or partial_stokes_vector):
                continue

            # Convert S0, S1, S2 from rgb to grayscale
            S0 = _to_gray(S0)
            S1 = _to_gray(S1)
            S2 = _to_gray(S2)

            if partial_stokes_vector:
                # save the s0 in the first color channel, etc...
                stokes_image = np.stack(
                    [
                        S0 * 8,
                        (S1 + (MAX_12BIT + 1)) * 8,
                        (S2 + (MAX_12BIT + 1)) * 8,
                    ],
                    axis=-1,
                )
                _write_rgb(os.path.join(cam_out, "partial_stokes_vector"), out_name, stokes_image)

            if not aolp_dolp:
                continue

            with np.errstate(divide="ignore", invalid="ignore"):
                dolp = np.sqrt(S1 ** 2 + S2 ** 2) / S0
            dolp = np.clip(np.nan_to_num(dolp, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
            aolp = np.arctan2(S2, S1) / 2  # in [-pi/2, pi/2]

            if aolp_encoding == "sine_and_cosine":
                # aolp is pi periodic, so sin/cos of twice the angle is the
                # continuous encoding: no wrap around discontinuity to learn
                aolp_r = (np.sin(2 * aolp) + 1) / 2
                aolp_g = (np.cos(2 * aolp) + 1) / 2
            else:
                aolp_r = aolp / np.pi + 0.5
                # aolp only has one channel here, so make the green channel 0
                aolp_g = np.zeros_like(aolp)

            # stack aolp and dolp channels
            aolp_dolp_image = np.stack([aolp_r, aolp_g, dolp], axis=-1) * 65535

            _write_rgb(os.path.join(cam_out, "aolp_dolp"), out_name, aolp_dolp_image)

        print(f"{raw_path} -> {out_name}")





if __name__ == "__main__":
    # Recomended output: oriented_intensities
    # 10-17-29_images
    convert_raw(
        input_path='../datasets/2026-09-07/10-17-29_images',
        output_path='./output/2026-09-07_10-17-29_images',
        oriented_intesities=True,
        partial_stokes_vector=False,
        color=False,
        aolp_dolp=False,
    )