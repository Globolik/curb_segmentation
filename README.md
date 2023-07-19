# Curb segmentation from point cloud
Scene L004.ply from [Toronto3D dataset](https://github.com/WeikaiTan/Toronto-3D) is used.

# Vis
![](utils/l4.png) 

# Method
1. Load .ply file
2. Sort point to get consecutive points
3. Select points that are close on XY plane and far on Z axis
4. From previous points select only those who have theta angle less that MAX_ANGLE and some elevation
   see [link](https://www.ri.cmu.edu/app/uploads/2019/06/FINAL-VERSION-TITS2018.pdf) for more details
5. Save result in .las file

# Requirements
- Python 3.8.10
- numpy
- plyfile
- pylas
- scipy
- torch==2.0.1
- tqdm

# Installation

    pip install -r requirements.txt

# Usage
```
usage: Road curb cloud segmentation [-h] [--output_path OUTPUT_PATH] input_file

positional arguments:
  input_file            path to .ply file

optional arguments:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        path to output .las file

```
