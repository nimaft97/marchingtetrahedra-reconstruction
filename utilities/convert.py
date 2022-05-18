import laspy
import numpy as np

if __name__ == '__main__':
    
    las = laspy.read('test.las')
    points = las.points
    
    print("meta data:\n\n")
    
    # all features that MAY be extracted
    print(f"all features that MAY be extracted:\n\t {list(las.point_format.dimension_names)}\n")
    
    # all features that are available
    print(f"all features that are available:\n\t {points.array.dtype}\n")
    
    # convert to numpy array and reduce the length if required
    points = np.array(points.array[:100]) # 100 is an arbitrary argument; if eliminated, the entire dataset will be converted
    
    # convert .las to .txt
    file = open("test.txt", "w+")
    np.savetxt(file, points)
    file.close() 
