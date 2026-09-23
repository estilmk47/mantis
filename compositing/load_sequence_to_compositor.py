# IMAGE_FOLDER_DATE = '2026-08-20'
IMAGE_FOLDER_TIME = '16-35-09'

RELATIVE_PATH_FROM_ROOT = f"../parser/output/{IMAGE_FOLDER_TIME}_images/cam0/oriented_intensities/"
INTESITY_AT_ANGLES = ('i000', 'i045', 'i090', 'i135')


IMAGE_SEQUENCE_NODES = ('i000_seq', 'i045_seq', 'i090_seq', 'i135_seq')

# Scene frame the first file of every sequence is shown on.
SEQUENCE_START_FRAME = 1
# Stretch the scene's frame range over the longest sequence that was loaded.
SET_SCENE_FRAME_RANGE = True


import os

import bpy
ROOT = bpy.data.filepath


def relative_to_abs_path(relative_path):
    """Resolve a path stated relative to the .blend file into an absolute one."""
    if not ROOT:
        raise RuntimeError("The .blend file is unsaved, so there is nothing to resolve relative paths against.")
    return os.path.normpath(bpy.path.abspath("//" + relative_path))


def compositor_node_tree(scene):
    """The scene's compositor node tree, on Blender 5.x as well as 4.x and older."""
    tree = getattr(scene, "compositing_node_group", None)  # Blender 5.x
    if tree is None:
        tree = getattr(scene, "node_tree", None)  # Blender <= 4.x
    if tree is None:
        raise RuntimeError(f"Scene '{scene.name}' has no compositor node tree.")
    return tree


def sequence_frames(folder):
    """The zero-padded .png files in *folder*, in playback order."""
    frames = sorted(name for name in os.listdir(folder) if name.lower().endswith(".png"))
    if not frames:
        raise RuntimeError(f"No .png frames found in {folder}")
    return frames


def find_sequence_node(tree, node_name, angle):
    """The image node to load *angle* into: by name, else by the folder it already points at."""
    node = tree.nodes.get(node_name)
    if node is not None:
        return node

    # The nodes ship unnamed (Image.004 ... Image.007), so fall back to whichever
    # image node is already a sequence reading from this angle's folder. The still
    # image nodes on the same angle are skipped because their source is 'FILE'.
    for candidate in tree.nodes:
        if candidate.bl_idname != "CompositorNodeImage" or candidate.image is None:
            continue
        if candidate.image.source != 'SEQUENCE':
            continue
        current = os.path.normpath(bpy.path.abspath(candidate.image.filepath))
        if os.path.basename(os.path.dirname(current)) == angle:
            return candidate

    raise KeyError(f"No compositor image node called '{node_name}', and no sequence node reading from an '{angle}' folder.")


def load_sequence_into_node(node, folder, frames):
    """Point *node* at the image sequence *frames* inside *folder*."""
    # Stored blend-relative, the way the .blend already references these folders.
    first_frame = bpy.path.relpath(os.path.join(folder, frames[0]))

    image = node.image
    if image is None:
        image = bpy.data.images.load(first_frame)
        node.image = image
    else:
        # Re-use the existing datablock so its colour space and alpha settings survive.
        image.filepath = first_frame
    image.source = 'SEQUENCE'
    image.reload()

    # Blender 5.x exposes the sequence settings on the node, older versions on its image user.
    image_user = getattr(node, "image_user", node)
    image_user.frame_duration = len(frames)
    image_user.frame_start = SEQUENCE_START_FRAME
    image_user.frame_offset = int(os.path.splitext(frames[0])[0]) - 1
    image_user.use_auto_refresh = True
    image_user.use_cyclic = False


if __name__ == "__main__":
    scene = bpy.context.scene
    tree = compositor_node_tree(scene)
    sequence_root = relative_to_abs_path(RELATIVE_PATH_FROM_ROOT)

    longest_sequence = 0
    for angle, node_name in zip(INTESITY_AT_ANGLES, IMAGE_SEQUENCE_NODES):
        folder = os.path.join(sequence_root, angle)
        frames = sequence_frames(folder)
        node = find_sequence_node(tree, node_name, angle)
        load_sequence_into_node(node, folder, frames)
        longest_sequence = max(longest_sequence, len(frames))
        print(f"{angle}: {len(frames)} frame(s) -> node '{node.name}' ({frames[0]} .. {frames[-1]})")

    if SET_SCENE_FRAME_RANGE:
        scene.frame_start = SEQUENCE_START_FRAME
        scene.frame_end = SEQUENCE_START_FRAME + longest_sequence - 1

    # Nudge the compositor so it re-evaluates with the new sequences.
    scene.frame_set(scene.frame_current)
    print(f"Loaded {len(INTESITY_AT_ANGLES)} sequences from {sequence_root}")
