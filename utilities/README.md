Data related utilities are provided for the purposes of processing and converting data into required formats.

It is highly recommended to use a Python virtual environment before running any of these scripts. Python and virtualenv are required and can be installed using Homebrew on Mac or a package manager on Linux. To create the virtual environment on Mac/Linux (Windows commands differ slightly):

```
python3 -m virtualenv /env
source env/bin/activate
pip install -r utilities/requirements.txt
```

The virtual environment can be deactivated using:

```
deactivate
```

The data-utils.py contains some functionality to process .las LiDAR point cloud files. To convert an .las file and output a comma-delimited .txt file use the following command:

```
cd utilities
python data-utils.py --file=../data/SOME_LIDAR_FILE.las --output ../data/output.txt
```

The --file argument is required but the --output argument is optional. Not including --output will result in no .txt file produced. The script also provides utilities to visualize the point cloud using Open3d as well as draw data plots.

_Note: By default, visualization and plots are turned off but can be turned back on from the entry function. In the future, these will be convertd to additional command line arguments for ease of use._