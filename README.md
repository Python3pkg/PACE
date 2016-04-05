# PACE

    |  _ \ / \  / ___| ____|
    | |_) / _ \| |   |  _|
    |  __/ ___ \ |___| |___
    |_| /_/   \_\____|_____|

PACE: Parameterization & Analysis of Conduit Edges
William Farmer - 2015

## Arguments

    usage: python3 linefit.py [-h] [-p] [-pd] [-a] [-mt] [-m MODEL] [-nnk NNK]
                              [-t TIME] [-d DATASTORE] [-pds] [-pdss]
                              [F [F ...]]

    Parameterize and analyze usability of conduit edge data

    positional arguments:
      F                     File(s) for processing. Each file has a specific
                            format: See README (or header) for specification.

    optional arguments:
      -h, --help            show this help message and exit
      -p, --plot            Create Plot of file(s)? Note, unless --time flag used,
                            will plot middle time.
      -pd, --plotdata       Create plot of current datastore.
      -a, --analyze         Analyze the file and determine Curvature/Noise
                            parameters. If --time not specified, will examine
                            entire file. This will add results to datastore with
                            false flags in accept field if not provided.
      -mt, --machinetest    Determine if the times from the file are usable based
                            on supervised learning model. If --time not specified,
                            will examine entire file.
      -m MODEL, --model MODEL
                            Learning Model to use. Options are ["nn", "svm",
                            "forest", "sgd"]
      -nnk NNK, --nnk NNK   k-Parameter for k nearest neighbors. Google it.
      -t TIME, --time TIME  Time (column) of data to use for analysis OR plotting.
                            Zero-Indexed
      -d DATASTORE, --datastore DATASTORE
                            Datastore filename override. Don't do this unless you
                            know what you're doing
      -pds, --printdata     Print data
      -pdss, --printdatashort
                            Print data short

## Datastore Format

    learning_data = [{filehash:[[trial_index, curvature, noise,
                                range, domain, accept, viscosity]
                        ,...],...},...]

The filehash is a SHA512 hash of the file contents read in 4kb chunks to prevent
hash collision.

New values can be added to this file manually, just take care that SHA512 is
used for these new key-value pairs.

## Incoming Data Format

`.mat` file, with the following format:

    {left_edges:2d_array,
     right_edges:2d_array,
     times:1d_array,
     accept:1d_array,
     viscosity:1d_array,
     ratio:float}

`left` and `right` edge arrays are 2 dimensional arrays where each column
corresponds to the time that that "photo" was taken. These are float values
corresponding to pixel heights.

The `time` array corresponds to the exact times that each column photo was
taken. These are float time values.

`accept` is an array that corresponds to whether or not each column was
accepted. These are integers 0 or 1. Any other values will be assumed to be 0.

`viscosity` is an array of viscosity values. Assumed to be 1 if not provided.

`ratio` is a float value that corresponds to the pixel-per-inch ratio.
