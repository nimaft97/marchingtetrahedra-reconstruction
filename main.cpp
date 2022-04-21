#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "reconstruction.h"

using namespace std;

int main(int argc, char *argv[])
{
    Eigen::Matrix<double, -1, 3> V; // vertices of the provided data
    Eigen::Matrix<int, -1, 3> F; // triangle definitions of the provided data
    Eigen::Matrix<double, -1, 3> N; // per-vertex normals of the provided data
    igl::readOFF("../data/bunny.off", V, F);
    igl::per_vertex_normals(V, F, N);

    vector<vector<double>> vertices;
    vector<vector<double>> normals;
    vector<vector<double>> faces;

    pair<vector<vector<double>>, vector<vector<int>> > R; // reconstruction vertices and faces

    mtr::matrix_to_2dvector<double, 3>(V, vertices);
    mtr::matrix_to_2dvector<double, 3>(N, normals);

    R = mtr::reconstruction<double>(vertices, normals);

}