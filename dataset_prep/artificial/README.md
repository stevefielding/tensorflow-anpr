# Deep ANPR

Using neural networks to build an automatic number plate recognition system.
See [this blog post](http://matthewearl.github.io/2016/05/06/cnn-anpr/) for an
explanation.

Usage is as follows:

1. `./extractbgs.py SUN397.tar.gz`: Extract ~3GB of background images from the [SUN database](http://groups.csail.mit.edu/vision/SUN/)
   into `bgs/`. (`bgs/` must not already exist.) The tar file (36GB) can be [downloaded here](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz).
   This step may take a while as it will extract 108,634 images.

2. `./gen_plates.py 1000`: Generate 1000 test set images in `CA_artificial/`. (`CA_artificial/` must not
    already exist.) This step requires `*.ttf` to be in the
    `fonts/` directory.

Different typefaces can be put in `fonts/` in order to match different type
faces.  With a large enough variety the network will learn to generalize and
will match as yet unseen typefaces. See
[#1](https://github.com/matthewearl/deep-anpr/issues/1) for more information.

