#include "marching_tets.h"

// mtr = marching tets recontruction
namespace mtr
{
    using namespace std;
    using namespace Eigen;

    template void generate_grid(MatrixXd &V, const RowVector3i &np, const RowVector3d &gmin, 
        const RowVector3d &gmax, const RowVector3d &pad);
    template void generate_tets_from_grid(MatrixXi &TF, const MatrixXd &TV, const RowVector3i &np);
    template vector<size_t> points_within_radius(const MatrixXd &P, const RowVector3d &origin, double radius);
    template RowVectorXd polynomial_basis_vector(int degree, const RowVector3d &p);
    template MatrixXd extract_rows(const MatrixXd &M, vector<size_t> indices); 
    template void compute_implicit_function_values(VectorXd &fx, const MatrixXd &TV, 
        const MatrixXd &C, const MatrixXd &D, double w);
    template void marching_tetrahedra(const MatrixXd &G, const MatrixXi &Tets,
        const VectorXd &fx, MatrixXd &SV, MatrixXi &SF);
    
    template <typename T>
    void generate_grid(T &V, const RowVector3i &np, const RowVector3d &gmin, 
        const RowVector3d &gmax, const RowVector3d &pad)
    {
        int nx = np(0);
        int ny = np(1);
        int nz = np(2);

        V = T::Zero(nx*ny*nz, 3);

        // Compute points on each coordinate
        ArrayXf x_points = ArrayXf::LinSpaced(nx, gmin(0) - pad(0), gmax(0) + pad(0));
        ArrayXf y_points = ArrayXf::LinSpaced(nx, gmin(1) - pad(1), gmax(1) + pad(1));
        ArrayXf z_points = ArrayXf::LinSpaced(nx, gmin(2) - pad(2), gmax(2) + pad(2));
        // Generate vertices of tet grid
        int vi = 0;
        for (int k = 0; k < z_points.rows(); k++)
        {
            for (int j = 0; j < y_points.rows(); j++)
            {
                for (int i = 0; i < x_points.rows(); i++)
                {
                    V.row(vi) = RowVector3d(x_points(i), y_points(j), z_points(k));
                    vi++;
                }
            }
        }
    }

    // NOTE: Move this function over to recon_utils
    template <typename T1, typename T2>
    vector<size_t> points_within_radius(const T1 &P, const T2 &origin, double radius)
    {
        VectorXd distances = (P.rowwise() - origin).rowwise().norm();
        vector<size_t> pi;
        for (int d = 0; d < distances.rows(); d++)
        {
            if (distances(d) < radius)
            {
                pi.push_back(d);
            }
        }
        return pi;
    }

    // TODO: Clean this up by using this function to call generate_grid, and move the parameters into here
    //   to reduce duplication
    template <typename T>
    void generate_tets_from_grid(MatrixXi &TF, const T &TV, const RowVector3i &np)
    {
        // each 'cube' will have 6 tets in it, and each tet is defined by 4 vertices
        TF = MatrixXi::Zero( (np(0)-1) * (np(1)-1) * (np(2)-1) * 6, 4);

        int nx = np(0);
        int ny = np(1);
        int nz = np(2);

        // keep track of which 'set' of tets we are on
        int ti = 0;

        // we will iterate over all 'cubes' in the grid and create 6 new tets
        for (int i = 0; i < nx - 1; i++)
        {
            for (int j = 0; j < ny - 1; j++)
            {
                for (int k = 0; k < nz - 1; k++)
                {
                    // collect the indices of the vertices of this cube
                    int v1 = (i+0)+nx*((j+0)+ny*(k+0));
                    int v2 = (i+0)+nx*((j+1)+ny*(k+0));
                    int v3 = (i+1)+nx*((j+0)+ny*(k+0));
                    int v4 = (i+1)+nx*((j+1)+ny*(k+0));
                    int v5 = (i+0)+nx*((j+0)+ny*(k+1));
                    int v6 = (i+0)+nx*((j+1)+ny*(k+1));
                    int v7 = (i+1)+nx*((j+0)+ny*(k+1));
                    int v8 = (i+1)+nx*((j+1)+ny*(k+1));

                    // form 6 tets using these vertices
                    TF.row(ti) << v1,v3,v8,v7;
                    TF.row(ti + 1) << v1,v8,v5,v7;
                    TF.row(ti + 2) << v1,v3,v4,v8;
                    TF.row(ti + 3) << v1,v4,v2,v8;
                    TF.row(ti + 4) << v1,v6,v5,v8;
                    TF.row(ti + 5) << v1,v2,v6,v8;
                    ti += 6; // increment our counter
                }
            }
        }
    }

    template <typename T>
    RowVectorXd polynomial_basis_vector(int degree, const T &p)
    {
        double x = p(0);
        double y = p(1);
        double z = p(2);

        RowVectorXd basis;

        if (degree == 0)
        {
            basis = RowVectorXd(1);
            basis << 1;
        }

        if (degree == 1)
        {
            basis = RowVectorXd(4);
            basis << 1, x, y, z;
        }

        if (degree == 2)
        {
            basis = RowVectorXd(10);
            basis << 1, x, y, z, x*y, x*z, y*z, x*x, y*y, z*z;
        }
        return basis;
    }
    
    template <typename T>
    MatrixXd extract_rows(const T &M, vector<size_t> indices)
    {
        MatrixXd P = MatrixXd::Zero(indices.size(), M.cols());

        // TODO: Is there a way to do this using matrix indices instead of using for loop??
        for (size_t i = 0; i < indices.size(); i++)
        {
            P.row(i) << M.row(indices[i]);
        }
        return P;
    }

    MatrixXd generate_weights_matrix(const MatrixXd &P, const RowVector3d &X, const double h)
    {
        // Welland weights matrix is a diagonal square matrix matching the size of input points
        MatrixXd W = MatrixXd::Zero(P.rows(), P.rows());
        for (int i = 0; i < P.rows(); i++)
        {
            double r = (P.row(i) - X).norm();
            double w = (4 * r/h + 1)*(1 - r/h);
            w = pow(w, 4);
            W(i, i) = w;
        }
        return W;
    }

    MatrixXd generate_basis_matrix(const MatrixXd &P)
    {
        // compute basis vector for first point to determine basis matrix size
        MatrixXd tmp = polynomial_basis_vector(2, P.row(0));
        MatrixXd B = MatrixXd::Zero(P.rows(), tmp.size());
        for (int i = 0; i < P.rows(); i++)
        {
            B.row(i) = polynomial_basis_vector(2, P.row(i));
        }
        return B;
    }

    template <typename T1, typename T2, typename T3, typename T4>
    void compute_implicit_function_values(T1 &fx, const T2 &TV, const T3 &C, const T4 &D, double w)
    {
        fx = VectorXd::Zero(TV.rows()); // one function value per grid point
        int min_num_pts = polynomial_basis_vector(2, TV.row(0)).size();

        // evaluate the implict function at each point in the tet grid
        for (int i = 0; i < TV.rows(); i++)
        {
            vector<size_t> pi = points_within_radius(C, TV.row(i), w);
            
            // assume this point is outside of the mesh
            if(pi.size() <= 0)
            {
                fx(i) = 100.0;
            }
            else
            {
                MatrixXd P = extract_rows(C, pi); // relevant constraint points
                VectorXd values = extract_rows(D, pi); // corresponding values
                
                MatrixXd W = generate_weights_matrix(P, TV.row(i), w);
                MatrixXd B = generate_basis_matrix(P);

                // Solve: (B.T*W*B)a = (B.T*W*D) 
                MatrixXd imd = B.transpose() * W;
                MatrixXd L;
                L = imd * B;
                MatrixXd R = imd * values;
                VectorXd a = L.llt().solve(R);
                a = L.ldlt().solve(R);

                // Calculate function value
                VectorXd gi = polynomial_basis_vector(2, TV.row(i));
                double v = gi.dot(a);
                fx(i) = v;
            } 
        }
    }

    RowVector3d GenerateTriangle(const RowVector3d &p1, const RowVector3d &p2,
        const double v1, const double v2, double snap)
    {
        if (abs(v1) < snap)
        {
            return p1;
        }
        else if (abs(v2) < snap)
        {
            return p2;
        }
        else if (abs(v2-v1) < snap)
        {
            return p1;
        }
        else
        {
            double mu = v2 / (v2 - v1);
            return (mu * p1) + ((1-mu) * p2);
        }
    }
    
    template <typename T1, typename T2, typename T3>
    void marching_tetrahedra(
        const T1 &G, // Tet grid vertices
        const MatrixXi &Tets, // Tets of the tet grid
        const T2 &fx, // implict function values at each grid vertex
        T3 &SV, // reconstructed mesh vertices
        MatrixXi &SF // reconstructed mesh faces
    )
    {
        vector<double> v_i; // holder for the mesh vertices we will build
        vector<int> f_i; // holder for the face definition
        int t = 0; // tri counter
        for (int i = 0; i < Tets.rows(); i++)
        {
            // march over each tet in our set of tets
            int config = 0; // the 'configuration' we have for the current tet

            // current tet
            RowVectorXi tet = Tets.row(i);

            // vertices of current tet
            RowVector3d p0 = G.row(tet(0));
            RowVector3d p1 = G.row(tet(1));
            RowVector3d p2 = G.row(tet(2));
            RowVector3d p3 = G.row(tet(3));

            // implict function values at current tet
            double v0 = fx(tet(0));
            double v1 = fx(tet(1));
            double v2 = fx(tet(2));
            double v3 = fx(tet(3));

            // Determine which type of configuration we have
            if (v0 < 0.0) {config += 1;}
            if (v1 < 0.0) {config += 2;}
            if (v2 < 0.0) {config += 4;}
            if (v3 < 0.0) {config += 8;}

            // Create tris based on the configuration we have
            switch (config)
            {
                case 0: case 15: // 0000 or 1111
                {
                    break;
                }
                case 1: 
                {
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p0, p2, v0, v2);
                    RowVector3d t3a = GenerateTriangle(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 2:
                {
                    RowVector3d t1a = GenerateTriangle(p1, p0, v1, v0);
                    RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    RowVector3d t3a = GenerateTriangle(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 3: 
                {
                    RowVector3d t1a = GenerateTriangle(p0, p3, v0, v3);
                    RowVector3d t2a = GenerateTriangle(p1, p2, v1, v2);
                    RowVector3d t3a = GenerateTriangle(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    //f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;

                    RowVector3d t1b = t2a;
                    RowVector3d t2b = t1a;
                    RowVector3d t3b = GenerateTriangle(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});
                    //f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 4: 
                {
                    // This is working correctly for both... figure out why and avoid duplication
                    RowVector3d t1a = GenerateTriangle(p2, p0, v2, v0);
                    RowVector3d t2a = GenerateTriangle(p2, p1, v2, v1);
                    RowVector3d t3a = GenerateTriangle(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 5: 
                {
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p0, p3, v0, v3);
                    RowVector3d t3a = GenerateTriangle(p1, p2, v1, v2);
                    //RowVector3d t1a = GenerateTriangle(p0, p3, v0, v3);
                    //RowVector3d t2a = GenerateTriangle(p2, p3, v2, v3);
                    //RowVector3d t3a = GenerateTriangle(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;

                    RowVector3d t1b = t2a;
                    RowVector3d t2b = GenerateTriangle(p2, p3, v2, v3);
                    RowVector3d t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                case 6: 
                {
                    //RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    //RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    //RowVector3d t3a = GenerateTriangle(p2, p3, v2, v3);
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    RowVector3d t3a = GenerateTriangle(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;

                    RowVector3d t1b = t3a;
                    RowVector3d t2b = t2a;
                    RowVector3d t3b = GenerateTriangle(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 7:
                {
                    RowVector3d t1a = GenerateTriangle(p3, p0, v3, v0);
                    RowVector3d t2a = GenerateTriangle(p3, p2, v3, v2);
                    RowVector3d t3a = GenerateTriangle(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                case 8:
                {
                    RowVector3d t1a = GenerateTriangle(p3, p0, v3, v0);
                    RowVector3d t2a = GenerateTriangle(p3, p2, v3, v2);
                    RowVector3d t3a = GenerateTriangle(p3, p1, v3, v1);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 9:
                {
                    //RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    //RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    //RowVector3d t3a = GenerateTriangle(p2, p3, v2, v3);
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    RowVector3d t3a = GenerateTriangle(p0, p2, v0, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;

                    RowVector3d t1b = t3a;
                    RowVector3d t2b = t2a;
                    RowVector3d t3b = GenerateTriangle(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                case 10:
                {
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p0, p3, v0, v3);
                    RowVector3d t3a = GenerateTriangle(p1, p2, v1, v2);
                    //RowVector3d t1a = GenerateTriangle(p0, p3, v0, v3);
                    //RowVector3d t2a = GenerateTriangle(p2, p3, v2, v3);
                    //RowVector3d t3a = GenerateTriangle(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;

                    RowVector3d t1b = t2a;
                    RowVector3d t2b = GenerateTriangle(p2, p3, v2, v3);
                    RowVector3d t3b = t3a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 11:
                {
                    // This is working correctly for both... figure out why and avoid duplication
                    RowVector3d t1a = GenerateTriangle(p2, p0, v2, v0);
                    RowVector3d t2a = GenerateTriangle(p2, p1, v2, v1);
                    RowVector3d t3a = GenerateTriangle(p2, p3, v2, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    //f_i.insert(f_i.end(), {t, t+1, t+2});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                case 12:
                {
                    RowVector3d t1a = GenerateTriangle(p0, p2, v0, v2);
                    RowVector3d t2a = GenerateTriangle(p1, p2, v1, v2);
                    RowVector3d t3a = GenerateTriangle(p1, p3, v1, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;

                    RowVector3d t1b = t3a;
                    RowVector3d t2b = GenerateTriangle(p0, p3, v0, v3);
                    RowVector3d t3b = t1a;
                    v_i.insert(v_i.end(), {t1b(0),t1b(1),t1b(2),t2b(0),t2b(1),t2b(2),t3b(0),t3b(1),t3b(2)});
                    f_i.insert(f_i.end(), {t, t+1, t+2});

                    t += 3;
                    break;
                }
                case 13:
                {
                    RowVector3d t1a = GenerateTriangle(p1, p0, v1, v0);
                    RowVector3d t2a = GenerateTriangle(p1, p3, v1, v3);
                    RowVector3d t3a = GenerateTriangle(p1, p2, v1, v2);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                case 14:
                {
                    RowVector3d t1a = GenerateTriangle(p0, p1, v0, v1);
                    RowVector3d t2a = GenerateTriangle(p0, p2, v0, v2);
                    RowVector3d t3a = GenerateTriangle(p0, p3, v0, v3);
                    v_i.insert(v_i.end(), {t1a(0),t1a(1),t1a(2),t2a(0),t2a(1),t2a(2),t3a(0),t3a(1),t3a(2)});
                    f_i.insert(f_i.end(), {t+2, t+1, t});

                    t += 3;
                    break;
                }
                default:
                {
                    break;
                }
            }
        }
    
        // Note: Using conservative matrix resizing is super slow, so we
        // simply create and fill a matrix using interm vectors
        SV = MatrixXd::Zero(v_i.size()/3, 3);
        SF = MatrixXi::Zero(f_i.size()/3,3);
        for(int i = 0; i < v_i.size(); i = i + 3)
        {
            SV.row(i/3) << v_i[i], v_i[i+1], v_i[i+2];
        }
        for (int i = 0; i < f_i.size(); i = i + 3)
        {
            SF.row(i/3) << f_i[i], f_i[i+1], f_i[i+2];
        }
    }

}
