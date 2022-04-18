#include "recon_utils.h"
#include "marching_tets.h"
#include "utils.h"
#include <igl/opengl/glfw/Viewer.h>
#include <igl/grid.h>
#include <igl/tetrahedralized_grid.h>
#include <igl/marching_tets.h>

int main(int argc, char *argv[])
{
  // Load Vertex and (optionall) tri data.
  // Currently we are using the .off file loader from IGL but this will need to 
  //  be replaced with functionality to load data from whatever is provided, 
  //  i.e. a .las file, raw data, etc.
  Eigen::MatrixXd V; // vertices of the provided data
  Eigen::MatrixXi F; // triangle definitions of the provided data
  igl::readOFF("../data/bunny.off", V, F);
  V = V * 25; // scale the data, we can probably get rid of this or use a heuristic
              // primarily here for better computation of welland weights
  V =  V.rowwise() - V.colwise().mean(); // re-center data on origin

  /*** 1. Compute per vertex normals ***/
  Eigen::MatrixXd N;
  igl::per_vertex_normals(V, F, N);  // TODO: THIS IS AN IGL FUNCTION, REPLACE IT
  // currently we are using igl's function, but we need to switch to a method we implement
  // ideally, normals data comes with our data and we don't need to do this step

  /*** 2. Compute constraints and values ***/
  Eigen::MatrixXd C; // implict function constraint points
  Eigen::MatrixXd D; // implict function values at corresponding constraints
  double eps = 0.01 * mtr::bounding_box_diagonal(V);
  mtr::generate_constraints_and_values(V, N, C, D, eps);

  /*** 3. Generate a tet grid ***/
  Eigen::RowVector3i num_tets = Eigen::RowVector3i(18,18,18); // number of tet points to generate in each direction
  Eigen::MatrixXd TV; // vertices of tet grid
  Eigen::MatrixXi TF; // tets of tet grid
  Eigen::RowVector3d gmin = V.colwise().minCoeff(); // grid minimum point
  Eigen::RowVector3d gmax = V.colwise().maxCoeff(); // grid maximum point
  Eigen::RowVector3d padding = Eigen::RowVector3d(eps, eps, eps); // additional padding for the grid
  mtr::generate_grid(TV, num_tets, gmin, gmax, padding);
  mtr::generate_tets_from_grid(TF, TV, num_tets);

  /*** 4. Compute implict function values at all tet grid points ***/
  Eigen::VectorXd fx;
  double welland_radius = 0.4; // TODO: try to define this programatically
  mtr::compute_implicit_function_values(fx, TV, C, D, welland_radius);

  /*** 5. March tets to create isosurface ***/
  Eigen::MatrixXd SV; // vertices of reconstructed mesh
  Eigen::MatrixXi SF; // faces of reconstructed mesh
  mtr::marching_tetrahedra(TV, TF, fx, SV, SF);

  /*** 6. Remove duplicate vertices ***/
  Eigen::MatrixXd SV2;
  mtr::merge_vertices(SV, SV2, SF, 0.001);

  /*** 7. Establish connected components from reconstructed mesh ***/
  /*** 8. Select largest connected component as reconstructed mesh ***/
  Eigen::MatrixXi SF2; 
  mtr::largest_connected_component(SV2, SF, SF2);

  igl::opengl::glfw::Viewer viewer;
  viewer.data().clear();
  viewer.data().set_mesh(SV2, SF2);
  viewer.launch();

  
  // draw stuff
  
  /*
  bool draw = false;

  if (draw)
  {
    // draw data
    // this code is temporary just for viewing the different steps/states of the algo
    // once we switch to UE, this will be replaced with UE's mesh generation calls
    igl::opengl::glfw::Viewer viewer;
    int current_tet = 0;
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
    {
      bool draw_axis = true;
      if(key == '1') // visualize original mesh
      {
        viewer.data().clear();
        viewer.data().set_mesh(V, F);
        viewer.data().set_face_based(true);
        return true;
      }
      else if (key == '2') // visualize original vertices only
      {
        viewer.data().clear();
        viewer.data().add_points(V, Eigen::RowVector3d(0.8, 0.8, 0.8));
        viewer.data().point_size = 12.0;
        return true;
      }
      else if (key == '3') // visualize constraint points and values
      {
        viewer.data().clear();
        for (int i = 0; i < V.rows(); i++)
        {
          viewer.data().add_points(C.row(i), Eigen::RowVector3d(0.8, 0.8, 0.8));
          viewer.data().add_points(C.row(V.rows() + i), Eigen::RowVector3d(0.0, 8.0, 0.0));
          viewer.data().add_points(C.row(V.rows()*2 + i), Eigen::RowVector3d(0.8, 0.0, 0.0));
          viewer.data().point_size = 8.0;
        }
      }
      else if (key == '4')
      {
        viewer.data().clear();
        viewer.data().add_points(TV, Eigen::RowVector3d(1.0, 0.7, 0.0));
        viewer.data().point_size = 10.0;
      }
      else if (key == '5')
      {
        // cycle through all tets so we can visualize and see where they are
        viewer.data().clear();
        viewer.data().add_points(TV, Eigen::RowVector3d(1.0, 0.7, 0.0));
        Eigen::RowVector4i t = TF.row(current_tet);
        Eigen::RowVector3d v1 = TV.row(t(0));
        Eigen::RowVector3d v2 = TV.row(t(1));
        Eigen::RowVector3d v3 = TV.row(t(2));
        Eigen::RowVector3d v4 = TV.row(t(3));
        viewer.data().add_edges(v1, v2, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().add_edges(v1, v3, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().add_edges(v1, v4, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().add_edges(v2, v3, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().add_edges(v2, v4, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().add_edges(v3, v4, Eigen::RowVector3d(1.0, 1.0, 1.0));
        viewer.data().point_size = 10.0;
        current_tet++;
        if (draw_axis)
        {
          // Draw axis points and edges
          viewer.data().add_points(Eigen::RowVector3d(0.0, 0.0, 0.0), Eigen::RowVector3d(1.0, 1.0, 1.0));
          viewer.data().add_points(Eigen::RowVector3d(1.0, 0.0, 0.0), Eigen::RowVector3d(1.0, 0.0, 0.0));
          viewer.data().add_points(Eigen::RowVector3d(0.0, 1.0, 0.0), Eigen::RowVector3d(0.0, 1.0, 0.0));
          viewer.data().add_points(Eigen::RowVector3d(0.0, 0.0, 1.0), Eigen::RowVector3d(0.0, 0.0, 1.0));
          viewer.data().add_edges(Eigen::RowVector3d(0.0, 0.0, 0.0), Eigen::RowVector3d(1.0, 0.0, 0.0), Eigen::RowVector3d(1.0, 0.0, 0.0));
          viewer.data().add_edges(Eigen::RowVector3d(0.0, 0.0, 0.0), Eigen::RowVector3d(0.0, 1.0, 0.0), Eigen::RowVector3d(0.0, 1.0, 0.0));
          viewer.data().add_edges(Eigen::RowVector3d(0.0, 0.0, 0.0), Eigen::RowVector3d(0.0, 0.0, 1.0), Eigen::RowVector3d(0.0, 0.0, 1.0)); 
        }
      }
      else if (key == '6')
      {
        viewer.data().clear();
        for (int i = 0; i < fx.rows(); i++)
        {
          if (fx(i) <= 0.0)
          {
            viewer.data().add_points(TV.row(i), Eigen::RowVector3d(1.0, 1.0, 1.0));
          }
        }
        viewer.data().point_size = 12.0;
      }
      else if (key == '7')
      {
        // draw reconstructed mesh
        viewer.data().clear();
        viewer.data().set_mesh(SV, SF);
        viewer.data().set_face_based(true);
      }
      else if (key == '8')
      {
        // draw largest connected component of reconstructed mesh
        viewer.data().clear();
        viewer.data().set_mesh(SV2, SF2);
        viewer.data().set_face_based(true);
      }
      else
      {
        return false;
      }
      return false;
    };
    viewer.launch();
  }

  */

}
