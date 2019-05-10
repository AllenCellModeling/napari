#### **_this repo is under development and not stable!_**

A fork of napari/napari intended for use at the Allen Institute for Cell Science. There is no support for this fork in any way shape or form, and instructions or documentation will differ from the official repo.

# napari

### multi-dimensional image viewer for python

[![License](https://img.shields.io/pypi/l/napari.svg)](https://github.com/napari/napari/raw/master/LICENSE)
[![Build Status](https://api.cirrus-ci.com/github/Napari/napari.svg)](https://cirrus-ci.com/napari/napari)
[![Python Version](https://img.shields.io/pypi/pyversions/napari.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/napari.svg)](https://pypi.org/project/napari)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/napari.svg)](https://pypistats.org/packages/napari)
[![Development Status](https://img.shields.io/pypi/status/napari.svg)](https://github.com/napari/napari)

**napari** is a fast, interactive, multi-dimensional image viewer for Python. It's designed for browsing, annotating, and analyzing large multi-dimensional images. It's built on top of `PyQt` (for the GUI), `vispy` (for performant GPU-based rendering), and the scientific Python stack (`numpy`, `scipy`).

We're developing **napari** in the open! But the project is in a **pre-alpha** stage. You can follow progress on this repository, test out new versions as we release them, and contribute ideas and code. Expect **breaking changes** from patch to patch.

## installation

**napari** can be installed on most Mac OS X and Linux systems with Python 3.6 or 3.7 by calling. 
```sh
$ conda create -n napari python=3.7
$ conda activate napari
$ git clone https://github.com/AllenCellModeling/napari.git
$ cd napari
$ pip install -e .
```


## HOW TO RUN

The code is set up to copy the files over to your computer first. __Please make sure you have 21 gb of space free for images before you start__. 

First start finder and press ⌘ + k to bring up the "Connect to Server" menu. Enter `smb://allen` and mount the `aics` drive. This allows us to automagically copy files over from the file system.

Then start the app. It should start copying over files for you. Once this is done, you should be able to disconnect from the network an annotate stuff remotely. Those results will be saved to your harddrive in the `./data/annotations folder`.

run 
```sh
$ python annotator.py
```

If you dont want to copy over files run for whatever reason do
```sh
$ python annotator.py --copy_to_local false
```

## features

Check out the scripts in the `examples` folder to see some of the functionality we're developing!

For example, you can add multiple images in different layers and adjust them

```python
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context

with app_context():
    # create the viewer with four layers
    viewer = ViewerApp(astronaut=rgb2gray(data.astronaut()),
                       photographer=data.camera(),
                       coins=data.coins(),
                       moon=data.moon())
    # remove a layer
    viewer.layers.remove('coins')
    # swap layer order
    viewer.layers['astronaut', 'moon'] = viewer.layers['moon', 'astronaut']
```

![image](resources/screenshot-layers.png)

You can add markers on top of an image

```python
from skimage import data
from skimage.color import rgb2gray
from napari import ViewerApp
from napari.util import app_context

with app_context():
    # setup viewer
    viewer = ViewerApp()
    viewer.add_image(rgb2gray(data.astronaut()))
    # create three xy coordinates
    points = np.array([[100, 100], [200, 200], [333, 111]])
    # specify three sizes
    size = np.array([10, 20, 20])
    # add them to the viewer
    markers = viewer.add_markers(points, size=size)
```

![image](resources/screenshot-add-markers.png)

**napari** supports bidirectional communication between the viewer and the Python kernel, which is especially useful in Jupyter notebooks -- in the example above you can retrieve the locations of the markers, including any additional ones you have drawn, by calling

```python
>>> markers.coords
[[100, 100],
 [200, 200],
 [333, 111]]
```

You can render and quickly browse slices of multi-dimensional arrays

```python

import numpy as np
from skimage import data
from napari import ViewerApp
from napari.util import app_context

with app_context():
    # create fake 3d data
    blobs = np.stack([data.binary_blobs(length=128, blob_size_fraction=0.05,
                                        n_dim=3, volume_fraction=f)
                     for f in np.linspace(0.05, 0.5, 10)], axis=-1)
    # add data to the viewer
    viewer = ViewerApp(blobs.astype(float))
```

![image](resources/screenshot-nD-image.png)

You can draw lines and polygons on an image, including selection and adjustment of shapes and vertices, and control over fill and stroke color. Run `examples/add_shapes.py` to generate and interact with the following example.

![image](resources/screenshot-add-shapes.png)

You can also paint pixel-wise labels, useful for creating masks for segmentation, and fill in closed regions using the paint bucket. Run `examples/labels-0-2d.py` to generate and interact with the following example.

![image](resources/screenshot-add-labels.png)

## plans

We're working on several features, including 

- support for 3D volumetric rendering
- support for multiple canvases
- a plugin ecosystem for integrating image processing and machine learning tools

See [this issue](https://github.com/napari/napari/issues/141) for some of the key use cases we're trying to enable, and feel free to add comments or ideas!

## contributing

Contributions are encouraged! Please read [our guide](https://github.com/napari/napari/blob/master/CONTRIBUTING.md) to get started. Given that we're in an early stage, you may want to reach out on [Github Issues](https://github.com/napari/napari/issues) before jumping in.
