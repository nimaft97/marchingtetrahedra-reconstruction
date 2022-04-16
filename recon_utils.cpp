#include "recon_utils.h"

namespace mtr 
{
    using namespace std;
    using namespace Eigen;

    template void print_matrix(const MatrixXd &M);
    template void print_matrix(const MatrixXi &M);
    template void print_matrix(const VectorXd &M);

    template double bounding_box_diagonal(const MatrixXd &V);
    template double bounding_box_diagonal(const MatrixXi &V);

    template pair<vector<size_t>, vector<double> >get_nearest_neighbours_with_distances(
        const MatrixXd &V, const RowVector3d &p, int n);

    template pair<RowVector3d, double> compute_constraint_and_value(
        const RowVector3d &v, const RowVector3d &n, double eps, double max_dist);

    template void generate_constraints_and_values(
        const MatrixXd &V, const MatrixXd &N, MatrixXd &C, MatrixXd &D, const double eps);

    template <typename T>
    void print_matrix(const T &M)
    {
        for (int i = 0; i < M.rows(); i++)
        {
            std::cout << "[";
            for (int j = 0; j < M.cols(); j++)
            {
            std::cout << M(i, j);
            if (j < M.cols() - 1)
            {
                std::cout << ", ";
            }
            }
            std::cout << "]" << std::endl;
        }
    }

    template <typename T>
    double bounding_box_diagonal(const T &V)
    {
        return (V.colwise().minCoeff() - V.colwise().maxCoeff()).norm();
    }

    bool contains(vector<size_t> vec,  int elem)
    {
        bool result = false;
        if(find(vec.begin(), vec.end(), elem) != vec.end() )
        {
            return true;
        }
        return false;
    }
    
    template <typename T1, typename T2>
    pair<vector<size_t>, vector<double> > get_nearest_neighbours_with_distances(const T1 &V, const T2 &p, int n)
    {
        vector<size_t> nn;
        vector<double> nn_dist;
        
        VectorXd distances = (V.rowwise() - p).rowwise().norm();

        // TODO: This is not the best way to do it, consider sorting the list using an insertion_sort 
        //  to partially sort the list
        // Find n minimums in distance vector
        while(nn.size() < n)
        {
            int min_idx = -1; // assume first idx is minimum
            double min_value = 100000000.0; // stupid hack, can't figure out why first point screws up
            for (int i = 0; i < V.rows(); i++)
            {
                if (!contains(nn, i) && min_idx != i && distances(i) > 0.0 && distances(i) < min_value)
                {
                    min_value = distances(i);
                    min_idx = i;
                }
            }
            nn.insert(nn.end(), min_idx);
            nn_dist.insert(nn_dist.end(), min_value);
        }
        return make_pair(nn, nn_dist);
    }
    
    template <typename T1, typename T2>
    pair<RowVector3d, double> compute_constraint_and_value(const T1 &v, const T2 &n, double eps, double max_dist)
    {
        RowVector3d c;
        double d;

        while (true)
        {
            c = v + n * eps;
            double dist = (c - v).norm();
            if (dist < max_dist)
            {
                d = (n * eps).norm();
                break; // success
            }
            else
            {
                eps /= 2; // failed criterion, reduce Epsilon and try again
            }
        }
        return make_pair(c, d);
    }
    
    template <typename T>
    void generate_constraints_and_values(const T &V, const T &N, T &C, T &D, const double eps)
    {
        // TODO: C and D should be generic i.e. Matrix<T, V.rows()*3, V.cols()>
        C = MatrixXd::Zero(V.rows()*3, V.cols()); // 3 constraints per point
        D = VectorXd::Zero(V.rows()*3, 1); // 1 constraint value per constraint point

        for (int i = 0; i < V.rows(); i++)
        {
            pair<vector<size_t>, vector<double> > nn = get_nearest_neighbours_with_distances(V, V.row(i), 1);
            
            // first set of constraints and points are the vertices themselves
            C.row(i) = V.row(i); // use data vertices as one set of constraints
            D(i) = 0; // implict function evaluates to 0 at the vertex

            // second set of constraints, points outside of the mesh
            pair<RowVector3d, double> c_d1 = compute_constraint_and_value(V.row(i), N.row(i), eps, nn.second[0]);
            C.row(V.rows() + i) = c_d1.first;
            D(V.rows() + i) = c_d1.second;

            // third set of constraints, points inside the mesh
            pair<RowVector3d, double> c_d2 = compute_constraint_and_value(V.row(i), -N.row(i), eps, nn.second[0]);
            C.row(V.rows()*2 + i) = c_d2.first;
            D(V.rows()*2 + i) = -c_d2.second;       
        }
    }
}