#pragma once

#include <iostream>
#include <vector>
#include <list>
#include <iterator>
#include <tuple>
#include <unordered_map>
#include <Eigen/Core>

namespace mtr
{
    using namespace std;
    using namespace Eigen;

    template <typename T1>
    void extract_rows(
        const T1 &M, // The matrix to extract rows from
        T1 &M2, // extract rows into this matrix
        vector<size_t> indices // indices to extract
    );

    template <typename T1, typename T2, typename T3>
    void replace_values(
        T1 &M, const vector<T2> &vals, 
        T3 new_val
    );

    template <typename T1, typename T2>
    void merge_vertices(
        const T1 &V, 
        T1 &V2, 
        T2 &F, 
        double eps
    );

    void print_adj_list(
        const unordered_map<int, vector<int> > &adj
    );

	void add_edge(
        vector<int>& vec, 
        int dst
    );

	void DFS(
        const unordered_map<int, vector<int> > &adj, 
        vector<bool> &visisted, 
        vector<int> &output, 
        int u
    );

	template <typename T1>
	unordered_map<int, vector<int>> adjacency_list(
        const T1& M
    );

	template <typename T1, typename T2>
	unordered_map<int, vector<int>> connected_components(
        const T1 &V, 
        const T2 &F
    );

    template <typename T1, typename T2>
    void largest_connected_component(
        const T1 &V, 
        const T2 &F, 
        T2 &F2
    );

} // namespace mtr
