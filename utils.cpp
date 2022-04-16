#include "utils.h"

namespace mtr
{
    using namespace std;
    using namespace Eigen;

    template void extract_rows(const MatrixXd &V, MatrixXd &V2, vector<size_t> indices);
    template void replace_values(MatrixXi &M, const vector<size_t> &vals, int val);
    template void merge_vertices(const MatrixXd &V, MatrixXd &V2, MatrixXi &F, double eps);
    template unordered_map<int, vector<int>> adjacency_list(const MatrixXi& M);
    template unordered_map<int, vector<int>> connected_components(const MatrixXd &V, const MatrixXi &F);
    template void largest_connected_component(const MatrixXd &V, const MatrixXi &F, MatrixXi &F2);

    template <typename T1>
    void extract_rows(const T1 &M, T1 &M2, vector<size_t> indices)
    {
        M2 = T1::Zero(indices.size(),M.cols());
        for (size_t i = 0; i < indices.size(); i++)
        {
            M2.row(i) = M.row(indices[i]);
        }
    }

    template <typename T1, typename T2, typename T3>
    void replace_values(T1 &M, const vector<T2> &vals, T3 new_val)
    { // could improve this by using a hash map instead, i.e. f_premerge => f_a
        for (int i = 0; i < M.rows(); i++)
        {
            for (int j = 0; j < M.cols(); j++)
            {
                // check if this value is contained in our vals
                if(find(vals.begin(), vals.end(), M(i,j)) != vals.end() )
                {
                    M(i,j) = new_val;
                }
            }
        }
    }

    template <typename T1, typename T2>
    void merge_vertices(const T1 &V, T1 &V2, T2 &F, double eps)
    {
        // kind of a hash table where index i represents vertex i, 
        // flag sets whether we have 'seen' this vertex
        vector<bool> seen = vector<bool>(V.rows(), false); 
        vector<size_t> merged;
        vector<RowVector3d> new_verts; // Eigen conservative resize is too heavy, this should be faster

        int current_new_v_i = 0;
        for (size_t i = 0; i < V.rows(); i++)
        {
            if(!seen[i]) // Note: Only process this vert if we haven't seen it yet!
            {
                // first build a temporary set of points from the vertices we SHOULD be searching...
                vector<size_t> indices;
                for (size_t j = 0; j < V.rows(); j++)
                {
                    if(!seen[j]) { indices.emplace_back(j);}
                }

                // now that we have the indices to look through, extract rows from V into temp
                T1 V_tmp;
                extract_rows(V, V_tmp, indices);
 
                // calculate distances from our current point to all points in V_tmp
                VectorXd d = (V_tmp.rowwise() - V.row(i)).rowwise().norm();
                vector<size_t> to_merge;
                for (size_t j = 0; j < d.size(); j++)
                {
                    if(d(j) < eps)
                    {
                        to_merge.emplace_back(indices[j]);
                    }
                }

                // merge vertices, add vertex to new set and update references in F to point to new vertex set index
                RowVector3d P = RowVector3d(0.0, 0.0, 0.0); // new point
                if (to_merge.size() > 0) // can't do division by zero
                {
                    for (size_t j = 0; j < to_merge.size(); j++)
                    {
                        P += V.row(to_merge[j]);
                        seen[to_merge[j]] = true;
                    }
                    P /= to_merge.size();
                    new_verts.emplace_back(P);
                    replace_values(F, to_merge, current_new_v_i);
                    current_new_v_i++;
                }
            }
        }
        // fill V2 with new verts
        V2 = T1::Zero(new_verts.size(), V.cols());
        for (int i = 0; i < new_verts.size(); i++)
        {
            V2.row(i) = new_verts[i];
        }
    }

    void print_adj_list(const unordered_map<int, vector<int> > &adj)
	{
		for (auto kv : adj)
		{
			cout << kv.first << ": ";
			for (auto v : kv.second)
			{
				cout << v << " ";
			}
			cout << endl;
		}
	}

    void add_edge(vector<int> &vec, int dst)
	{
		if (find(vec.begin(), vec.end(), dst) == vec.end())
		{
			vec.push_back(dst);
		}
	}  

    template <typename T1>
	unordered_map<int, vector<int>> adjacency_list(const T1 &M)
	{
		unordered_map<int, vector<int> > adj;
		for (int i = 0; i < M.rows(); i++)
		{
			int v1 = M(i, 0);
			int v2 = M(i, 1);
			int v3 = M(i, 2);

			if (adj.find(v1) == adj.end()) { adj.insert({ v1, vector<int>() }); }
			if (adj.find(v2) == adj.end()) { adj.insert({ v2, vector<int>() }); }
			if (adj.find(v3) == adj.end()) { adj.insert({ v3, vector<int>() }); }

			add_edge(adj[v1], v2);
			add_edge(adj[v1], v3);
			add_edge(adj[v2], v1);
			add_edge(adj[v2], v3);
			add_edge(adj[v3], v1);
			add_edge(adj[v3], v2);
		}
		return adj;
	}

    void DFS(const unordered_map<int, vector<int> > &adj, vector<bool> &visisted, vector<int> &output, int u)
	{
		visisted[u] = true;
		output.push_back(u);

		for (int i = 0; i < adj.at(u).size(); i++)
		{
			int uu = adj.at(u)[i];
			if (!visisted[uu])
			{
				DFS(adj, visisted, output, uu);
			}
		}
	}

    template <typename T1, typename T2>
	unordered_map<int, vector<int>> connected_components(const T1 &V, const T2 &F) 
	{
		unordered_map<int, vector<int> > adj = adjacency_list(F); // construct adjacency list of connected edges
        unordered_map<int, vector<int>> cc;
		vector<bool> seen(V.rows(), false); // hold which items we have seen so far

		// perform depth first search on each key in the adj_list
        int idx = 0;
		for (auto kv : adj)
		{
			vector<int> output;
			DFS(adj, seen, output, kv.first);
			if (output.size() > 1) // only interested on connected components with more than one vertex
			{
                cc.insert({idx, output});
                idx++;
			}
		}
        return cc;
	}

    template <typename T1, typename T2>
    void largest_connected_component(const T1 &V, const T2 &F, T2 &F2)
    {
        unordered_map<int, vector<int>> cc = connected_components(V, F);
        int largest_cc_size = 0;
        int largest_cc_key;
        for (auto kv : cc)
        {
            if(kv.second.size() > largest_cc_size)
            {
                largest_cc_key = kv.first;
                largest_cc_size = kv.second.size();
            }
        }
        
        vector<bool> keep(F.rows(), false); // store the faces we want to keep for quick lookup
        vector<RowVector3i> new_F; // temporary holder for the faces we want to keep

        // flag the faces we want to keep
        for (int f : cc[largest_cc_key])
        {
            keep[f] = true;
        }

        for (int i = 0; i < F.rows(); i++)
        { // note: to keep a face, all three vertices must show up in our keep map
            if(keep[F(i,0)] && keep[F(i,1)] && keep[F(i,2)])
            {
                new_F.emplace_back(F.row(i));
            }
        }

        F2 = MatrixXi::Zero(new_F.size(), 3);
        for (int i = 0; i < new_F.size(); i++)
        {
            F2.row(i) = new_F[i];
        }
    }
} // namespace mtr


