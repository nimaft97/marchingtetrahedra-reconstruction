#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <Eigen/Core>

// mtr = marching tets reconstruction
namespace mtr 
{
    using namespace std;
    using namespace Eigen;

    template <typename T>
    void print_matrix(
        const T &M // matrix/vector to print
    );

    template <typename T>
    double bounding_box_diagonal(
        const T &V // set of points to compute bounding box diagonal size for
    );

    bool contains(
        const std::vector<size_t> vec, // vector containing indices to check
        int elem // element to look for
    );

    template <typename T1, typename T2>
    pair<vector<size_t>, vector<double> > get_nearest_neighbours_with_distances(
        const T1 &V, // The set of vertices to compute distances to
        const T2 &p, // The point of origin for distance computation
        int n // the number of neighbours to find
    );

    template <typename T1, typename T2>
    pair<RowVector3d, double> compute_constraint_and_value(
        const T1 &v, // a vertex to compute for
        const T2 &n, // corresponding normal for the vertex
        double eps, // starting epsilon distance scaler
        double max_dist // computed point must be closer than this
    );

    template <typename T>
    void generate_constraints_and_values(
        const T &V, // vertex data
        const T &N, // per vertex normals
        T &C, // constraint points to fill
        T &D, // constraint values to fill
        const double eps // 'thickness' of area for implict function computation
    );
    
}

