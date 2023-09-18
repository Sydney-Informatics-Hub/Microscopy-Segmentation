# Cell Segment Toolkit 

The cell segmentation toolkit is a collection of tools that can be used to convert data from one format to another. The toolkit is designed to be used in conjunction with 3D cell frameworks such as IMOD, MONAI, 3DSclicer, Avizo, and AI-assisted segmentation annotation tools such as Anylabeling and COCO format.

## Installation

The toolkit is written in Python and can be installed using pip:

```bash
conda create -n cellsegment python=3.10
conda activate cellsegment
conda env update --file environment.yaml
```

## Overview

The following conversion tools have been built as part of this project:

```mermaid

graph TD
    anylabel[Anylabeling: .json]
    imod[IMOD: .mod]
    coco[COCO: .json]
    mesh[MESH: OBJ, VRML, STL, PLY, OFF, GLB, COLLADA]
    csv[CSV: .csv]
    tiff[TIFF: .tiff]
    avizo[Avizo: .am]

    imod-- imod2coco.py -->coco
    imod-- imod2csv.py -->csv 
    imod-- imod2labels.py -->anylabel
    imod-- imod2mesh.py -->mesh 
    anylabel-- labels2imod.py -->imod
    anylabel-- anylabeling2tif.py -->tiff
    tiff-- Avizo import -->avizo

```
