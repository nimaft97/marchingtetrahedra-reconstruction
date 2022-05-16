import numpy as np
import laspy

if __name__ == '__main__':
	xyz_file = laspy.read('test.las').xyz # it can be limitted to certain rows, laspy.read('test.las').xyz[:N]
	file = open("test.txt", "w+")
	np.savetxt(file, xyz_file)
	file.close()
	


