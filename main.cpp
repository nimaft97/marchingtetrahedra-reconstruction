#include <igl/opengl/glfw/Viewer.h>
#include "reconstruction.h"

using namespace std;

int main(int argc, char *argv[])
{
    // read data from .off file
    Eigen::Matrix<double, -1, 3> V; // vertices of the provided data
    Eigen::Matrix<int, -1, 3> F; // triangle definitions of the provided data
    Eigen::Matrix<double, -1, 3> N; // per-vertex normals of the provided data
    igl::readOFF("../data/bunny.off", V, F);
    igl::per_vertex_normals(V, F, N);

    // collect data into containers
    std::vector<std::vector<double>> vertices;
    std::vector<std::vector<double>> normals;
    std::vector<std::vector<double>> faces;

    // convert data to STL vectors
    mtr::matrix_to_2dvector<double, 3>(V, vertices);
    mtr::matrix_to_2dvector<double, 3>(N, normals);

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<int>> > R;
    R = mtr::reconstruction<double>(vertices, normals, 15, 15, 15, 200.0, 4.0, 0.01);

    // reconstructed mesh in Eigen matrices
    Eigen::Matrix<double, -1, 3> V2;
    Eigen::Matrix<int, -1, 3> F2;

    // convert data into Eigen matrices for plotting
    mtr::vector2d_to_matrix(R.first, V2);
    mtr::vector2d_to_matrix(R.second, F2);

    // test reconstruction
    igl::opengl::glfw::Viewer viewer;
    viewer.data().clear();
    viewer.data().set_mesh(V2, F2);
    viewer.launch();
}