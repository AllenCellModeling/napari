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
from napari import ViewerApp
from napari.util import app_context
from os.path import isfile
import warnings
from vispy.color import Colormap

import aicsimageio.cziReader as cziReader


def get_default_range(image, mode):

    if mode == "fluor":
        lb = np.percentile(image, 50)
        hb = np.percentile(image, 99.99)
    if mode == "bf":
        lb = np.percentile(image, 0.5)
        hb = np.percentile(image, 99.5)

    return lb, hb


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
        annotations_layer = viewer.add_labels(labels, name="annotations", opacity=0.75)
    else:
        annotations_layer = viewer.layers["annotations"]
        annotations_layer.image = labels

    annotations_layer.n_dimensional = False


skimage_save_warning = "'%s is a low contrast image' % fname"

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=skimage_save_warning
    )

## Preliminary testing. Change this later
im_dir = "./data/"
save_dir = "./annotations/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

image_paths = glob("{}/*.czi".format(im_dir))
annotation_paths = [
    "{}/annotation_{}.tiff".format(save_dir, os.path.basename(image_path))
    for image_path in image_paths
]

curr_index = 0


def get_index():
    return curr_index


def set_index(index):
    global curr_index

    if index < 0:
        index = 0
    if index >= len(image_paths):
        index = len(image_paths) - 1

    curr_index = index


with app_context():
    # create an empty viewer
    viewer = ViewerApp()

    # add the first image
    load_image(viewer, image_paths[curr_index], annotation_paths[curr_index])

    def save(viewer, layer_name="annotations"):
        """Save the current annotations
        """
        labels = viewer.layers[layer_name].image.astype(np.uint16)
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

        msg = "Loading " + viewer.layers[0].name
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
        viewer.status = msg

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
