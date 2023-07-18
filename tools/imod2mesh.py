"""
Convert IMOD model to suitable mesh formats.

Mesh output formats supported:
OBJ, STL, binary PLY, OFF, GLB, COLLADA, JSON, DICT

Requirements:
------------
- IMOD installed (tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- trimesh (pip install trimesh[easy])
- pycollada (For COLLADA export)


Example
-------
import imod2mesh
infname = 'IMOD_model.mod'
mesh = imod2mesh.convert_mesh(infname)
"""

import os
import subprocess
import trimesh

def convert_mesh(infname, 
     output_format = 'OBJ',
     args_imod2obj = '',
     return_mesh = True):
    """
    Reads in IMOD model (.mod) or other mesh models and converts to desired mesh format.

    Parameters
    ----------
    infname : str
        Path to IMOD model file (.mod) or mesh file.
    output_format : str 
        'OBJ' (.obj), 
        'STL' (.stl), 
        'PLY' (.ply), 
        'OFF' (.off), 
        'GLB' (.glb), 
        'COLLADA' (.dae),
        'JSON' (.json),
        'DICT' (.dict)
    args_imod2obj : str
        Additional arguments for imod2obj command. See IMOD documentation for details.
        (https://bio3d.colorado.edu/imod/doc/man/imod2obj.html)
    return_mesh : bool
        If True, returns mesh as trimesh object. If False, returns None.

    Returns
    -------
    mesh : trimesh
    """

    if output_format == 'OBJ':
        outfname = infname.replace('.mod', '.obj')
    elif output_format == 'STL':
        outfname = infname.replace('.mod', '.stl')
    elif output_format == 'PLY':
        outfname = infname.replace('.mod', '.ply')
    elif output_format == 'OFF':
        outfname = infname.replace('.mod', '.off')
    elif output_format == 'GLB':
        outfname = infname.replace('.mod', '.glb')
    elif output_format == 'COLLADA':
        outfname = infname.replace('.mod', '.dae')
    elif output_format == 'JSON':
        outfname = infname.replace('.mod', '.json')
    elif output_format == 'DICT':
        outfname = infname.replace('.mod', '.dict')
    else:
        print(f'Output format {output_format} not supported. Please choose from: OBJ, STL, PLY, OFF, GLTF, COLLADA, JSON, DICT')
        return None

    # if IMOD model, need to convert first to obj
    if infname.endswith('.mod'):
        # create temporary file to save obj file
        outfname_obj = infname.replace('.mod', '.obj')

        # run imod model to mesh obj conversion
        imod_cmd = f'imod2obj {infname} {outfname_obj} {args_imod2obj}'
        try:
            print('Converting IMOD model to mesh...')
            subprocess.run([imod_cmd], shell = True, check = True)
        except subprocess.CalledProcessError:
            print('Error with imod2obj command:' + imod_cmd)
            return None

        # read in obj file as trimesh
        mesh = trimesh.load(outfname_obj)

    else:
        mesh = trimesh.load(infname)

    # save mesh in output_format
    if output_format != 'OBJ':
        mesh.export(outfname)#, file_type = output_format.lower())

    if return_mesh:
        return mesh
    else:
        return None
