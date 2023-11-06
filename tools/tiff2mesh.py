# sconverting array to point cloud and saving as mesh

import numpy as np
import pyvista as pv
import meshio
import argparse

def arrays_to_point_cloud(arrays):
    """
    Convert a stack of 2D binary numpy arrays into a 3D point cloud.

    :param arrays: A 3D numpy array where each 2D slice along the third axis is a binary mask.
    :return: A numpy array of points.
    """
    points = np.argwhere(arrays)  # Get the indices of points where the array is not zero
    return points

def point_cloud_to_mesh(points):
    """
    Convert a point cloud to a mesh using Delaunay 3D triangulation.

    :param points: A numpy array of points.
    :return: A PyVista mesh object.
    """
    cloud = pv.PolyData(points)
    volume = cloud.delaunay_3d(alpha=1.0)
    surface = volume.extract_surface()  # Extract the surface from the UnstructuredGrid
    return surface

def save_mesh(mesh, filename, file_format = 'obj'):
    """
    Save a mesh as a VRML file using meshio.

    :param mesh: A PyVista mesh object.
    :param filename: The output file path.
    """
    cells = [("triangle", mesh.faces.reshape(-1, 4)[:, 1:])]
    meshio.write_points_cells(
        filename,
        mesh.points,
        cells,
        file_format=file_format
    )

def main(array_stack, filename, file_format="obj"):
    points = arrays_to_point_cloud(array_stack)
    surface = point_cloud_to_mesh(points)
    save_mesh(surface, filename, file_format)

# Test function to create synthetic data and test the pipeline
def test_conversion():
    # Create synthetic data: a stack of 3 10x10 arrays with a simple shape
    arrays = np.zeros((10, 10, 3), dtype=np.uint8)
    arrays[3:7, 3:7, :] = 1  # Create a cube in the center

    # Convert arrays to point cloud
    points = arrays_to_point_cloud(arrays)
    
    # Convert point cloud to mesh
    surface = point_cloud_to_mesh(points)
    
    # Save mesh as VRML
    vrml_filename = 'synthetic_data_mesh.obj'
    save_mesh(surface, vrml_filename, file_format='obj')

    return vrml_filename

def main(array_stack, filename, file_format="vrml"):
    points = arrays_to_point_cloud(array_stack)
    surface = point_cloud_to_mesh(points)
    save_mesh(surface, filename, file_format)

# write main execution with argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a stack of 2D binary numpy arrays to a 3D mesh.')
    parser.add_argument('array_stack', type=str, help='The path to the input stack of 2D binary numpy arrays.')
    parser.add_argument('filename', type=str, help='The name of the output mesh file.')
    parser.add_argument('--file_format', type=str, default='obj', help='The file format of the output mesh file.')
    args = parser.parse_args()
    main(args.array_stack, args.filename, args.file_format)
