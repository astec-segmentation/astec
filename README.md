# ASTEC

This file contains instructions for installing and running the ASTEC reconstruction algorithm.

ASTEC is a **Segmentation and Tracking algorithm from Contact-dependent cell communications drive morphological invariance during ascidian embryogenesis**.

This work was originaly published on **bioRxiv** in 2018:

**Contact-dependent cell-cell communications drive morphological invariance during ascidian embryogenesis**, _Léo Guignard, Ulla-Maj Fiuza, Bruno Leggio, Emmanuel Faure, Julien Laussu, Lars Hufnagel, Grégoire Malandain, Christophe Godin, Patrick Lemaire_, bioRxiv 2018; doi: https://doi.org/10.1101/238741.

It was later published in **Science** in 2020:

**Contact area–dependent cell communication and the morphological invariance of ascidian embryogenesis**, _Léo Guignard, Ulla-Maj Fiuza, Bruno Leggio, Julien Laussu, Emmanuel Faure, GAËL Michelin, Kilian Biasuz, Lars Hufnagel, Grégoire Malandain, Christophe Godin, Patrick Lemaire_, Science 2020; DOI: https://doi.org/10.1126/science.aar5663.


## I - Contents of the repository

The folder contains the following elements:
- definitions.py : the definitions of the folder process and the RAWDATA parameters (time / angle etc..)
- ASTEC  : the folder  containing all the libraries 
- 1-fuse.py 
- 2-mars.py
- 3-manualcorrection.py        
- 4-astec.py
- 5-postcorrection.py
- 6-named.py
- 7-virtualembryo.py 
- checklineage.py : tools for check lineage safety 
- rename.py : tools to rename file 
- README.md
- license.txt   : the licence terms you accept by using the workflow

        
## II - Installation and software requirements

### Installation

#### System dependencies
The Python package `PyLibtiff` require the `libtiff*-dev` system library.
On Ubuntu it can be installed with:
```
sudo apt-get install libtiff5-dev
```

#### Conda package
To be released soon.

#### From sources in a conda environment
You first have to clone the ASTEC project from its INRIA GitLab repository:
```bash
git clone https://gitlab.inria.fr/morpheme/astec.git
```

You may now use the conda environment recipe available in `pkg/env/`:
```bash
cd astec
conda env create -f pkg/env/astec.yaml
```

**Any missing dependency to add here ??!**

Finally, you can install the sources in the activated `astec` environment:
```bash
conda activate astec
python -m pip install -e .
```

**Notes**: the `-e` option install the package in "editable" mode, this is want you want if you aim at contributing to the ASTEC project. 


## III - Running ASTEC

### Previously :
Different manual steps are required to process the workflow

- Copy the entire folder to the dataset folder 
- update definitions.py with this dataset information only the first lines have to be updated such as :
```
# RAWDATA DEFINITION
begin=4 #Stating point
end=95 #Last Point

delta = 1 # Delta between two time points (if one does not want to fuse every single time point)
ori = 'right' # if im2 angle - im1 angle < 0 => right
resolution = (.17, .17, 1.) # Resolution of the raw images
delay = 0 # If the time stamps in the folder are not the actual time stamps in the global movie
mirrors = False  #TO COMMENT
target_resolution = .3 # Isotropic resolution of the final fused image
```
- Check the raw data file named 
- Launch the fusion : python 1-fusion.py
- Launch the segmentation of the first time step : python 2-mars.py
- Check the file in FUSE/SEG/<EN>_fuse_seg_t<time_begin>_mars.tiff to fuse oversegmentation and maps them in the file 3-manualcorrection.py 
- Launch the manual segmentation : python 3-manualcorrection.py        
- Launch the global segmentation (very long) : python 4-astec.py
- Launch the post segmentation : python 5-postcorrection.py
- Create a file with the correspond names with the first time step in  media/DATA/<<EN>>/<EN>-names.txt
	with this format : <pixel id (from tiff file)>:<cell name> 
- Launch the named : python 6-named.py
- Upload the embryon in 4DCloudEmbryon with : python 7-virtualembryo.py 


### From version v2.0 :
Data architecture 
```
<EMBRYO>
    RAWDATA
        LC
            Stack0000
            Stack0001
        RC
            Stack0000
            Stack0001
    FUSE
        FUSE_<EXP_1>
        FUSE_<EXP_2>
        ...
        FUSE_RELEASE
    SEG
        SEG_<EXP>
        ...
        SEG_RELEASE
    POST
        POST_<EXP>
        ...
        POST_RELEASE
```
"RELEASE" sub-directories should content the "last version" validated by an expert.

Particular files:
    nomenclature.py : file fixing the set of naming rules in working directories
      -> this file should not be modified
    parameters.py : file defining the set of parameters useful for all the process steps (parameters are all prefixed with respect to the step they are used in).
      -> it is a "template" file which should be duplicated and whose copy can be modified by the user to its convenience (only the parameter values should be modified, not their name...).

The scripts of steps 1-fuse.py, 2-mars.py, 3-manualcorrection.py, 4-astec.py, 5-postcorrection.py are executables, so that each astec step calling can be made as described here:
For example, in order to launch the fusion step on an embryo called "171107-Karine-St8", one should:

    Duplicate the file <astec-package>/parameters.py -> new file <arbitrary-path>/parameters_karine.py
    Edition of file <arbitrary-path>/parameters_karine.py to specify the desired value of each parameter related to the fusion step
    In a terminal,
        $ cd <astec-package> # in order to be in the astec directory (/media/DATA/Codes/astec-package on Hermione)
        $ ./1-fuse.py --parameters  <arbitrary-path>/parameters_karine.py --embryo-rep /media/DATA/171107-Karine-St8/
        (or equivalently,a shorter format)
        $ ./1-fuse.py -p  <arbitrary-path>/parameters_karine.py -e /media/DATA/171107-Karine-St8/

At each astec step execution, a copy of the parameters file as well as a log file are automatically generated in the target working directory.

For each astec step, it is possible to display the help relative to the corresponding script by launching the script with the option '--help'. For example, for the fusion step:

    In a terminal, launch the command line:
        $ <astec-package>/1-fuse.py --help
    The terminal displays the following message:

    Usage: 1-fuse.py [options]

    Options:
      -h, --help            show this help message and exit
      -p FILE, --parameters=FILE
                                python file containing parameters definition
      -e PATH, --embryo-rep=PATH
                                path to the embryo data
      -q, --quiet          don't print status messages to stdout


## IV - Building conda packages

### Build pylibtiff
```
conda build external/libtiff-0.4.0/pkg/ --user morpheme
```

### Build astec
```
conda build pkg/recipe/ -c morpheme -c conda-forge --user morpheme
```