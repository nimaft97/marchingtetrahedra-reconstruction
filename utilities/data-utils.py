import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import laspy, os, sys

def load_las_file(filename) -> tuple[np.ndarray, laspy.LasHeader]:
    file_path = os.path.join(os.getcwd(), filename)
    try:
        las = laspy.read(file_path)
        print("File successfully loaded!")
        return np.vstack((las.x,las.y, las.z)).transpose(), las.header
    except FileExistsError as fee:
        sys.exit(f"No file found at {file_path}")
    except Exception as e:
        sys.exit(f"Could not open file, {e}")

def save_to_file(points, normals, outfile, delimiter=",", mode="w"):
    d = delimiter
    try:
        print(f"Saving output to: {outfile}...")
        with open(outfile, mode=mode) as outfile:
            for p, n in zip(points, normals):
                outfile.write(f"{p[0]}{d}{p[1]}{d}{p[2]}{d}{n[0]}{d}{n[1]}{d}{n[2]}{d}")
        print(f"Done.")
    except Exception as e:
        sys.exit(f"Could not write to file at {outfile}, {e}")

def preprocess_data(data, scale=None, offset=None, recenter=None, rotate=None, crop=None) -> np.ndarray:
    data2 = np.copy(data)
    # Note: These are not commutative!
    if scale is not None:
        data2 *= scale
        if offset is not None:
            data2 -= offset * scale

    # Apply a rotational matrix
    if rotate is not None:
        # Using o3d as its easier than implementing a rotational matrix
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data2)
        R = pcd.get_rotation_matrix_from_xyz((rotate[0], rotate[1], rotate[2]))
        pcd.rotate(R)
        data2 = np.asarray(pcd.points)

    # These operations are commutative
    if recenter:
        data2 = data2 - np.average(data2, axis=0)

    # Crop last
    if crop is not None:
        # Compute range of data
        mins = np.min(data2, axis=0)
        r = np.max(data2, axis=0) - mins

        # Create masks for each dimension
        for i in range(data.shape[1]):
            m1 = data2[:,i] > mins[i] + (r[i] * crop[i])
            m2 = data2[:,i] < mins[i] + (r[i] * crop[i+3])
            mask = np.logical_and(m1, m2)
            data2 = data2[mask]

    return data2

def generate_point_cloud(data, estimate_normals=False, visualize=False):
    # Use o3d to produce point cloud visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    if estimate_normals:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # TODO: Enhance this part by using the bbox idea to reorient normals
    if visualize:
        o3d.visualization.draw_geometries([pcd])
    return pcd

def plot_data_stats(data, display_pt_limit=300000, plt_title=""):
    # Figure out what interval to display points at
    if data.shape[0] > display_pt_limit:
        smt = int(data.shape[0] / display_pt_limit)
    else:
        smt = display_pt_limit

    data2 = data[::smt]

    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches(12,12)

    fig.suptitle(f"LiDAR Dataset Statistics {plt_title}", fontsize=16)

    ax[0,0].set_title(f"Point distribution in Z ({data.shape[0]} points) ")
    ax[0,0].hist(data[:, 2], bins=1000, orientation="horizontal")
    ax[0,0].set_xlabel("Count")
    ax[0,0].set_ylabel("Z value bin (1000 bins)")

    ax[0,1].set_title(f"Z-value of points {data2.shape[0]} points)")
    ax[0,1].scatter(data2[:,0], data2[:,1], c=data2[:,2], s=0.1)
    ax[0,1].set_xlabel("x value")
    ax[0,1].set_ylabel("y value")

    ax[1,0].set_title(f"X-Z point distribution {data2.shape[0]} points)")
    ax[1,0].scatter(data2[:,0], data2[:,2], s=0.1)
    ax[1,0].set_xlabel("x value")
    ax[1,0].set_ylabel("z value")

    ax[1,1].set_title(f"Y-Z point distribution {data2.shape[0]} points)")
    ax[1,1].scatter(data2[:,1], data2[:,2], s=0.1)
    ax[1,1].set_xlabel("y value")
    ax[1,1].set_ylabel("z value")

    fig.tight_layout()
    plt.show()


def main(file, output=None, visualize=False, plot_stats=False):
    # Load data
    data, head = load_las_file(file)

    # Preprocess data
    scale = np.array([head.x_scale, head.y_scale, head.z_scale])
    offset = np.array([head.x_offset, head.y_offset, head.z_offset])
    rotate = np.array([0.0 , 0.0 , 0.35 * np.pi/4]) # rotate in x, y, z
    
    # find Z min and max independently using percentiles
    x_min, x_max, y_min, y_max = 0.20, 0.45, 0.3, 0.55
    z_min = 0.99*(np.quantile(data, 0.01, axis=0)[2] - np.min(data[:,2]))/np.ptp(data[:,2])
    z_max = 1.01*(np.quantile(data, 0.99, axis=0)[2] - np.min(data[:,2]))/np.ptp(data[:,2])
    crop = np.array([x_min, y_min, z_min, x_max, y_max, z_max]) # percentages
    data = preprocess_data(data, scale=scale, offset=offset, recenter=True, crop=crop)

    # Convert data to point cloud
    pcd = generate_point_cloud(data, estimate_normals=True, visualize=visualize)

    if plot_stats:
        plot_data_stats(data, display_pt_limit=300000, plt_title=file.split("/")[-1])

    if output:
        save_to_file(np.asarray(pcd.points), np.asarray(pcd.normals), 
            outfile=os.path.join(os.getcwd(), output), delimiter=",", mode="w")

if __name__ == "__main__":
    arg_dict = {"file": None, "output": None}
    args = [a for a in sys.argv[1:]]
    if len(args) == 0:
        sys.exit("Error: No arguments provided. At minimum, a file to process is required, i.e. --file=FILENAME")
    for a in args:
        p, v = a.split("=")
        if p[2:] not in arg_dict.keys():
            sys.exit("Error: Incorrect arguments provided. Required format is: --file=FILENAME --output=OUTFILE_NAME")
        else:
            arg_dict[p[2:]] = v
    if arg_dict["file"] is None:
        sys.exit("Error: A file to process must be provided, i.e. --file=FILENAME")
    
    main(file=arg_dict["file"], output=arg_dict["output"], visualize=False, plot_stats=False)