"""
Modified from
https://github.com/sofroniewn/image-demos/blob/master/examples/kaggle_nuclei_editor.py
and
https://github.com/sofroniewn/image-demos/blob/master/examples/allen_cell.py
"""

import numpy as np
from skimage.io import imread, imsave
import os
from glob import glob
from os.path import isfile
import warnings
from vispy.color import Colormap
from natsort import natsorted
import argparse
import pandas as pd
import shutil
import tqdm

import aicsimageio.cziReader as cziReader

from napari import ViewerApp
from napari.util import app_context
import napari.layers._labels_layer._constants as layer_constants


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(description="Annotator for GE/SCE")
parser.add_argument(
    "--start_from_last_annotation",
    default=True,
    type=str2bool,
    help="Start from last annotation",
)
parser.add_argument(
    "--save_if_empty",
    default=False,
    type=str2bool,
    help="Save annotation layer if it is empty",
)
parser.add_argument("--data_dir", type=str, default="./data/")
parser.add_argument(
    "--copy_to_local", type=str2bool, default=True, help="Copy images to local first"
)

args = parser.parse_args()
print(args)

df = pd.read_csv("{}/napari_annotation_files.csv".format(args.data_dir))


def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


if args.copy_to_local:
    im_dir = "{}/images/".format(args.data_dir)
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)

    image_paths = np.array(
        [
            "./data/images/{}".format(os.path.basename(file_path))
            for file_path in df.file_path
        ]
    )

    for i in tqdm.tqdm(range(len(df.file_path))):

        if not os.path.exists(image_paths[i]):
            shutil.copyfile(
                df.file_path[i].replace("/allen/", "/Volumes/"), image_paths[i]
            )

else:
    image_paths = np.array(
        [file_path.replace("/allen/", "/Volumes/") for file_path in df.file_path]
    )


ref_files = image_paths[df.set == "reference"]
annotate_files = image_paths[df.set == "annotate"]
np.random.shuffle(annotate_files)

image_paths = np.concatenate([ref_files, annotate_files])


save_dir = "{}/save_dir/".format(args.data_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


annotation_paths = [
    "{}/annotation_{}.tiff".format(save_dir, os.path.basename(image_path))
    for image_path in image_paths
]

if not args.start_from_last_annotation:
    curr_index = 0
else:
    missing_files = ~np.array(
        [os.path.exists(annotation_path) for annotation_path in annotation_paths]
    )
    if not np.any(missing_files):
        curr_index = len(annotation_paths) - 1
    else:
        curr_index = np.where(missing_files)[0][0]


def get_default_range(image, mode):

    if mode == "fluor":
        lb = np.percentile(image, 50)
        hb = np.percentile(image, 99.99)
    if mode == "bf":
        lb = np.percentile(image, 0.5)
        hb = np.percentile(image, 99.5)

    return lb, hb


skimage_save_warning = "'%s is a low contrast image' % fname"

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=skimage_save_warning
    )


def get_max_index():
    return len(image_paths)


def get_index():
    return curr_index


def set_index(index):
    global curr_index

    if index < 0:
        index = 0
    if index >= get_max_index():
        index = get_max_index() - 1

    curr_index = index


with app_context():
    # create an empty viewer
    viewer = ViewerApp()

    def save(viewer, layer_name="annotations"):
        """Save the current annotations
        """
        labels = viewer.layers[layer_name].image.astype(np.uint16)

        if (not args.save_if_empty) and np.all(labels == 0):
            msg = "{} layer is empty. save_if_empty set to False.".format(
                viewer.layers[layer_name].name
            )
            print(msg)
            viewer.status = msg
            return

        save_path = annotation_paths[curr_index]
        imsave(save_path, labels, plugin="tifffile", photometric="minisblack")
        msg = "Saving " + viewer.layers[layer_name].name + ": " + save_path
        print(msg)
        viewer.status = msg

    def next(viewer):
        """Save the current annotation and load the next image and annotation
        """
        save(viewer)

        set_index(get_index() + 1)
        load_image(viewer, image_paths[get_index()], annotation_paths[get_index()])

        msg = "Loading " + image_paths[curr_index]
        print(msg)
        viewer.status = msg

    def previous(viewer):
        """Save the current annotation and load the previous image and annotation
        """
        save(viewer)

        set_index(get_index() - 1)
        load_image(viewer, image_paths[get_index()], annotation_paths[get_index()])

        msg = "Loading " + image_paths[curr_index]
        print(msg)
        viewer.status = msg

    def revert(viewer, layer_name="annotations"):
        """Loads the last saved annotation
        """
        if isfile(annotation_paths[curr_index]):
            labels = imread(annotation_paths[curr_index])
        else:
            labels = np.zeros(viewer.layers[layer_name].image.shape, dtype=np.int)

        viewer.layers[layer_name].image = labels

        msg = "Reverting " + viewer.layers[layer_name].name
        print(msg)
        # viewer.status = msg

    def increment_label(viewer, layer_name="annotations"):
        """Increments current label
        """
        label = viewer.layers[layer_name].selected_label
        viewer.layers[layer_name].selected_label = label + 1

    def decrement_label(viewer, layer_name="annotations"):
        """Decrements current label
        """
        label = viewer.layers[layer_name].selected_label
        if label > 0:
            viewer.layers[layer_name].selected_label = label - 1

    def background_label(viewer, layer_name="annotations"):
        """Set current label to background
        """
        viewer.layers[layer_name].selected_label = 0

    def max_label(viewer, layer_name="annotations"):
        """Sets label to max label in visible slice
        """
        label = viewer.layers[layer_name]._image_view.max()
        viewer.layers[layer_name].selected_label = label + 1

    def load_image(viewer, im_path, im_labels_path):
        with cziReader.CziReader(im_path) as reader:
            cells = reader.load()[0]

        layer_names = [layer.name for layer in viewer.layers]

        ch_nums = [1, 2, 3, 4, 0]
        ch_names = ["red spots", "structure", "yellow spots", "DNA", "brightfield"]
        ch_types = ["fluor", "fluor", "fluor", "fluor", "bf"]
        ch_colors = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]

        for ch_num, ch_name, ch_type, ch_color in zip(
            ch_nums, ch_names, ch_types, ch_colors
        ):
            if ch_name not in layer_names:
                ch = viewer.add_image(cells[:, ch_num, :, :], name=ch_name)
            else:
                ch = viewer.layers[ch_name]
                ch.image = cells[:, ch_num, :, :]

            ch.colormap = Colormap([(0, 0, 0, 1), ch_color])
            ch.clim = get_default_range(cells[:, ch_num, :, :], ch_type)
            ch.blending = "additive"

        # for this case, annotations are only 2D
        if os.path.exists(im_labels_path):
            labels = imread(im_labels_path)
        else:
            labels = np.zeros(cells[0, 0, :, :].shape, dtype=np.int)

        if "annotations" not in layer_names:
            annotations_layer = viewer.add_labels(
                labels, name="annotations", opacity=0.75
            )
        else:
            annotations_layer = viewer.layers["annotations"]
            annotations_layer.image = labels

        annotations_layer.n_dimensional = False

        if annotations_layer.mode == layer_constants.Mode.FILL:
            annotations_layer.mode = layer_constants.Mode.PAINT
            viewer.status = "Switched to paint mode for your safety."

        max_label(viewer)

        display_string = "{}/{}: {}".format(get_index(), get_max_index(), im_path)
        # msg = "Saving " + viewer.layers[layer_name].name + ": " + save_path
        # print(msg)
        # viewer.status = display_string
        viewer.title = display_string

    # add the first image
    load_image(viewer, image_paths[curr_index], annotation_paths[curr_index])

    custom_key_bindings = {
        "s": save,
        "r": revert,
        "n": next,
        "b": previous,
        "i": increment_label,
        "m": max_label,
        "d": decrement_label,
        "t": background_label,
    }

    viewer.key_bindings = custom_key_bindings
