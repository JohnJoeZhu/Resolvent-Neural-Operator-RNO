import os
import numpy as np
import torch
import torch.optim as optim
from geometry import get_operators
import nibabel as nib

def read_surf(fname):
    """
    Read FreeSurfer's surface.

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    vertex_coords : 2D array, shape = [n_vertex, 3]
        Vertex coordinates.
    faces : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    TRIANGLE_FILE_MAGIC_NUMBER = 0xFFFFFE
    QUAD_FILE_MAGIC_NUMBER = 0xFFFFFF
    NEW_QUAD_FILE_MAGIC_NUMBER = 0xFFFFFD

    with open(fname, "rb") as f:
        h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
        magic = (h0 << 16) + (h1 << 8) + h2

        if (magic == QUAD_FILE_MAGIC_NUMBER) | (magic == NEW_QUAD_FILE_MAGIC_NUMBER):
            # need to be verified
            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            vnum = (h0 << 16) + (h1 << 8) + h2

            h0, h1, h2 = np.fromfile(f, dtype=np.dtype("B"), count=3)
            fnum = (h0 << 16) + (h1 << 8) + h2

            vertex_coords = np.fromfile(f, dtype=np.dtype(">i2"), count=3 * vnum) / 100
            vertex_coords = vertex_coords.reshape(-1, 3)
            arr = np.fromfile(f, dtype=np.dtype("B"), count=9 * fnum)
            faces = (arr[0::3] << 16) + (arr[0::3] << 8) + arr[0::3]
            faces = faces.reshape(-1, 3)

            return vertex_coords, faces

        elif magic == TRIANGLE_FILE_MAGIC_NUMBER:
            f.readline()
            f.readline().strip()

            vnum, fnum = np.fromfile(f, dtype=np.dtype(">i4"), count=2)
            vertex_coords = np.fromfile(f, dtype=np.dtype(">f4"), count=3 * vnum)
            faces = np.fromfile(f, dtype=np.dtype(">i4"), count=3 * fnum)

            vertex_coords = vertex_coords.reshape(vnum, 3)
            faces = faces.reshape(fnum, 3)

            return vertex_coords, faces

        else:
            raise Exception("SurfReaderError: unknown format!")

def read_vtk(fname):
    """
    Read a vtk file (ASCII version).

    Parameters
    __________
    fname : str
        File path.

    Returns
    _______
    v : 2D array, shape = [n_vertex, 3]
        3D coordinates of the the input mesh.
    f : 2D array, shape = [n_face, 3]
        Triangles of the input mesh.
    """

    with open(fname, "rb") as fd:
        lines = iter(l for l in fd)

        ver = next(d for d in lines if b"Version" in d)
        ver = float(ver.split()[-1])

        nVert = next(d for d in lines if b"POINTS" in d)
        nVert = int(nVert.split()[1])
        v = np.fromfile(fd, dtype=float, count=nVert * 3, sep=" ").reshape(nVert, 3)

        nFace = next(d for d in lines if b"POLYGONS" in d)
        nFace = int(nFace.split()[1])
        if ver < 5:
            f = np.fromfile(fd, dtype=int, count=nFace * 4, sep=" ").reshape(nFace, 4)
            f = f[:, 1:]
        else:
            nFace -= 1
            next(d for d in lines if b"CONNECTIVITY" in d)
            f = np.fromfile(fd, dtype=int, count=nFace * 3, sep=" ").reshape(nFace, 3)

    return v, f

def read_annot(fname):
    """
    Read FreeSurfer annotation file and convert to one-hot encoding.
    
    Parameters
    __________
    fname : str
        File path to .annot file.
        
    Returns
    _______
    labels_one_hot : 2D array, shape = [n_vertex, n_classes]
        One-hot encoded labels.
    label_names : list
        Names of the labels.
    """
    # Read annotation file using nibabel
    labels, ctab, names = nib.freesurfer.read_annot(fname)
    
    # Decode label names
    label_names = [name.decode('utf-8') for name in names]
    
    # Convert to one-hot encoding
    n_vertices = len(labels)
    n_classes = len(label_names)
    
    # Create one-hot encoding
    labels_one_hot = np.zeros((n_vertices, n_classes), dtype=np.float32)
    for i, label in enumerate(labels):
        if label >= 0 and label < n_classes:  # Valid label index
            labels_one_hot[i, label] = 1.0
    
    return labels, label_names


def load_surface_with_labels(white_path, annot_path):
    """
    Load surface geometry and corresponding labels.
    
    Parameters
    __________
    white_path : str
        Path to white surface file.
    annot_path : str
        Path to annotation file.
        
    Returns
    _______
    data_dict : dict
        Dictionary containing vertices, faces, labels, and label names.
    """
    # Read surface geometry
    vertices, faces = read_surf(white_path)
    
    # Read labels
    labels, label_names = read_annot(annot_path)
    
    # Convert to PyTorch tensors
    vertices = torch.from_numpy(vertices.astype(np.float32))
    faces = torch.from_numpy(faces.astype(np.int32))
    labels = torch.from_numpy(labels.astype(np.float32))
    
    # Create data dictionary
    data_dict = {
        "vertices": vertices,
        "faces": faces,
        "labels": labels,
        "label_names": label_names
    }
    
    return data_dict

def visualize_surface_with_labels(vertices, faces, labels, label_names=None, 
                                  title="Surface with Labels", figsize=(10, 8)):
    """
    Simple visualization of surface with labels.
    
    Parameters
    __________
    vertices : array, shape = [n_vertex, 3]
        Vertex coordinates.
    faces : array, shape = [n_face, 3]
        Triangle faces.
    labels : array, shape = [n_vertex, n_classes] or [n_vertex]
        Labels for each vertex.
    label_names : list, optional
        Names of the labels.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size.
    """
    # Convert to numpy if needed
    if torch.is_tensor(vertices):
        vertices = vertices.numpy()
    if torch.is_tensor(faces):
        faces = faces.numpy()
    if torch.is_tensor(labels):
        labels = labels.numpy()
    
    # If labels are one-hot, convert to class indices
    if labels.ndim == 2 and labels.shape[1] > 1:
        class_indices = np.argmax(labels, axis=1)
    else:
        class_indices = labels
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    
    # Color by labels
    scatter = ax.scatter(x, y, z, c=class_indices, cmap='tab20', s=1, alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Label Classes')
    
    # Adjust view
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print label statistics
    unique_labels, counts = np.unique(class_indices, return_counts=True)
    print(f"Label Statistics:")
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(class_indices)) * 100
        if label_names and label < len(label_names):
            name = label_names[label]
        else:
            name = f"Label_{label}"
        print(f"  {name}: {count} vertices ({percentage:.2f}%)")

data_dir = "data/cortex_sample"
label_dir = "data/cortex_sample_labels"
subjs = os.listdir(data_dir)
for subj in subjs:
    if "." in subj:
        continue
    white_dir = os.path.join(data_dir, subj, "surf", "lh.white")
    parc_dir = os.path.join(data_dir, subj, "label", "lh.labels.DKT31.manual.annot")
    data_dict = load_surface_with_labels(white_dir, parc_dir)
    print(f"  Vertices: {data_dict['vertices'].shape}")
    print(f"  Faces: {data_dict['faces'].shape}")
    print(f"  Labels: {data_dict['labels'].shape}")
    print(f"  Number of classes: {data_dict['label_names']}")
    data_dict['vertices'] = torch.from_numpy(data_dict['vertices'].astype(np.float32))
    data_dict['faces'] = torch.from_numpy(data_dict['faces'].astype(np.int32))
    # frames, massvec, L, evals, evecs, dis_norm=get_operators(data_dict['vertices'], data_dict['faces'], k_eig=128,op_cache_dir="data/cortex_sample/op_cache")


    # white_v, white_f = read_surf(white_dir)
    # white_v = torch.from_numpy(white_v.astype(np.float32))
    # white_f = torch.from_numpy(white_f.astype(np.int32))
    # print(white_v.shape, white_f.shape)
    # # _, mass, L, evals, evecs, gradX, gradY = compute_operators(white_v, white_f)
    # label = torch.from_numpy(np.loadtxt(parc_dir).astype(np.int16))
    # d = dict()
    # d["vertices"] = white_v
    # d["label"] = label
    # # d["massvec"] = mass
    # # d["evals"] = evals
    # # d["evecs"] = evecs
    # # d["gradX"] = gradX
    # # d["gradY"] = gradY
    # torch.save(d, os.path.join("/data/lfs/kimjongmin8/Develop/CortexDiffusion/Mindboggle_dataset",subj+".pt"))