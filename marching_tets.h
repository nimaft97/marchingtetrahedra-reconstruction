#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace mtr
{
    using namespace std;
    using namespace Eigen;

    template <typename T>
    void generate_grid(
        T &V, // vertices data structure to fill
        const RowVector3i &np, // number of points in each (x, y, z) direction to generate
        const RowVector3d &gmin, // grid minimum point
        const RowVector3d &gmax, // grid maximum point
        const RowVector3d &pad // padding to add to grid
    );

    template <typename T>
    void generate_tets_from_grid(
        MatrixXi &TF, // tet definitions to fill in
        const T &TV, // grid vertices to use
        const RowVector3i &np // number of points in each direction of the grid
    );

    // NOTE: Move this function over to recon_utils
    template <typename T1, typename T2>
    vector<size_t> points_within_radius(
        const T1 &P, // set of points to search in
        const T2 &origin, // origin point to compute distances from
        double radius // the ball radius to look within
    );

    template <typename T>
    RowVectorXd polynomial_basis_vector(
        int degree, // degree of polynomial
        const T &p // the point to compute the basis vector for
    );

    template <typename T> 
    MatrixXd extract_rows(
        const T &M, // matrix to extract rows from
        vector<size_t> indices // row indices to extract
    );

    MatrixXd generate_weights_matrix(
        const MatrixXd &P, // relevant points to compute for
        const RowVector3d &X, // tet grid point to compute distances from
        const double h // height of the welland function
    );

    MatrixXd generate_basis_matrix(
        const MatrixXd &P // the points to compute basis matrix for
    );

    template <typename T1, typename T2, typename T3, typename T4>
    void compute_implicit_function_values(
        T1 &fx, // per point implict function values to fill
        const T2 &TV, // the tet vertices to compute values at
        const T3 &C, // constraint points to use in calculations
        const T4 &D, // values at constraint points
        double w // welland radius for point effect decay
    );

    RowVector3d GenerateTriangle(const RowVector3d &p1, const RowVector3d &p2,
        const double v1, const double v2, double snap = 0.01);

    template <typename T1, typename T2, typename T3>
    void marching_tetrahedra(
        const T1 &G, // Tet grid vertices
        const MatrixXi &Tets, // Tets of the tet grid
        const T2 &fx, // implict function values at each grid vertex
        T3 &SV, // reconstructed mesh vertices
        MatrixXi &SF // reconstructed mesh faces
    );

    // why can't this be defind here??
    //template vector<size_t> points_within_radius(const MatrixXd &P, const RowVector3d &origin, double radius);
}