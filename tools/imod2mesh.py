"""
Convert IMOD model to suitable mesh formats.
The generated mesh file is saved in the same directory as the input file
and can be used for 3D visualization in e.g. Meshlab, Blender, etc.

The following mesh output formats are supported:
OBJ, VRML/VRML2, STL, PLY, OFF, GLB, COLLADA

Usage:
python imod2mesh.py -i IMOD_model.mod -o OBJ

Required Arguments:
-i, --input: Filename for IMOD model file
-o, --output_format: Output format, default is OBJ

Optional Arguments:
-a, --args_imod2obj: Additional arguments for imod2obj command. See IMOD documentation for details.

Installation Requirements:
-------------------------
- IMOD (tested with IMOD 4.12), see for installation instructions:
    https://bio3d.colorado.edu/imod/download.html
- Python 3.8+
- trimesh (pip install trimesh[easy])
- pycollada (For COLLADA export)


Python Example
--------------
import imod2mesh
infname = 'IMOD_model.mod'
mesh = imod2mesh.convert_to_mesh(infname, output_format = 'STL')


Author: Sebastian Haan, The University of Sydney, 2023
"""

import os
import argparse
import subprocess
import trimesh

def convert_to_mesh(infname, 
     output_format = 'OBJ',
     args_imod2obj = ''):
    """
    Reads IMOD model (.mod) or other mesh models and converts to desired mesh format.

    Parameters
    ----------
    infname : str
        Path to IMOD model file (.mod) or mesh file.
    output_format : str 
        'OBJ' (.obj), 
        'VRML' or 'VRML2' (.wmt),
        'STL' (.stl), 
        'PLY' (.ply), 
        'OFF' (.off), 
        'GLB' (.glb), 
        'COLLADA' (.dae)
    args_imod2obj : str
        Additional arguments for imod2obj command. See IMOD documentation for details.
        (https://bio3d.colorado.edu/imod/doc/man/imod2obj.html)

    Returns
    -------
    mesh : trimesh
    """

    if output_format == 'OBJ':
        outfname = infname.replace('.mod', '.obj')
    elif (output_format == 'VRML') | (output_format == 'VRML2'):
        outfname = infname.replace('.mod', '.wrl')
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
        print(f'Output format {output_format} not supported.\
        Please choose from: OBJ, VRML, VRML2, STL, PLY, OFF, GLB, COLLADA, JSON, DICT')
        return None

    if (output_format == 'VRML') & (infname.endswith('.mod')):
        imod_cmd = f'imod2vrml {infname} {outfname}'
        try:
            print('Converting IMOD model to VRML mesh...')
            subprocess.run([imod_cmd], shell = True, check = True)
        except subprocess.CalledProcessError:
            print('Error with imod2vrml command:' + imod_cmd)
            return None
        print('VRML mesh saved as ' + outfname)
        return None
    
    if (output_format == 'VRML2') & (infname.endswith('.mod')):
        imod_cmd = f'imod2vrml2 -a {infname} {outfname}'
        try:
            print('Converting IMOD model to VRML2 mesh...')
            subprocess.run([imod_cmd], shell = True, check = True)
        except subprocess.CalledProcessError:
            print('Error with imod2vrml2 command:' + imod_cmd)
            return None
        print('VRML2 mesh saved as ' + outfname)
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
    print('Mesh saved as ' + outfname)
    return None

def main():
    """ Main Function """
    parser = argparse.ArgumentParser(description='Convert IMOD model to suitable mesh formats.')
    parser.add_argument('-i', '--input', help='Input IMOD model file (.mod) or mesh file', required=True)
    parser.add_argument('-o', '--output_format', help='Output format', default='OBJ', choices=['OBJ', 'VRML', 'VRML2', 'STL', 'PLY', 'OFF', 'GLB', 'COLLADA'])
    parser.add_argument('-a', '--args_imod2obj', help='Additional arguments for imod2obj command', default='')
    args = parser.parse_args()

    convert_to_mesh(args.input, args.output_format, args.args_imod2obj)

if __name__ == '__main__':
    main()
