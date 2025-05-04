#include <iostream>
#include <sstream>
#include <fstream>
#include <string>

#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <vector>
#include <functional>

#include <constants.h>
#include <rbf_interpolator.hpp> 

using namespace std;
using namespace Eigen;

const PRECISION_TYPE mm_to_m = 1e-3;
const PRECISION_TYPE mass = 1.67e-27; // Mass of a hydrogen atom in kilograms
const PRECISION_TYPE bohr_magneton = 9.274e-24; // Bohr magneton in J/T
const PRECISION_TYPE mu = bohr_magneton/mass; // Magnetic moment per unit mass

const PRECISION_TYPE grid_step = 0.00295309; // Grid step in meters
const PRECISION_TYPE half_grid_step = grid_step / 2.0; // Half grid step


// The initial Loading point from which the atoms are launched in the simulation
const Vector3d loding_pt = Vector3d(22.996172, 22.996172, 201.506241) * mm_to_m;


void load_csv(const string &filename,
    vector<Vector3d> &positions,
    vector<PRECISION_TYPE> &Bx_vals,
    vector<PRECISION_TYPE> &By_vals,
    vector<PRECISION_TYPE> &Bz_vals)
{

ifstream file(filename);
if (!file.is_open())
{
cerr << "Failed to open file: " << filename << endl;
exit(1);
}

string line;

// Skip header line
getline(file, line);

while (getline(file, line))
{
// if (line.empty())
//     cout << "Empty line encountered, skipping." << endl;
// continue;
stringstream ss(line);
string val;
int col_indx = 0;

// Read
while (getline(ss, val, ','))
{
  if (col_indx < 3)
  {
      if (col_indx == 0)
      {
          positions.emplace_back(Vector3d());
          positions.back().x() = stold(val);
      }
      else if (col_indx == 1)
      {
          positions.back().y() = stold(val);
      }
      else if (col_indx == 2)
      {
          positions.back().z() = stold(val);
      }
  }
  else if (col_indx == 3)
  {
      Bx_vals.push_back(stold(val));
  }
  else if (col_indx == 4)
  {
      By_vals.push_back(stold(val));
  }
  else if (col_indx == 5)
  {
      Bz_vals.push_back(stold(val));
  }
  col_indx++;
}
}

file.close();
cout << "Loaded " << positions.size() << " data points." << endl;
}

double distance(const Vector3d &p1, const Vector3d &p2)
{
return (p1 - p2).norm();
}

void extract_points_within_radius(const MatrixXd &X,
                        const VectorXd &Bx,
                        const VectorXd &By,
                        const VectorXd &Bz,
                        MatrixXd &X_near,
                        VectorXd &Bx_near,
                        VectorXd &By_near,
                        VectorXd &Bz_near,
                        const Vector3d &query_point, double r_tol)
{
vector<int> selected_indices;

for (int i = 0; i < X.cols(); ++i)
{
// Check if the point is within the radius of the query point
Vector3d point = X.col(i);
if (distance(point, query_point) <= r_tol)
{
  selected_indices.push_back(i);
}
}

// Now extract the selected points and their corresponding B-field values
int num_selected = selected_indices.size();
X_near.resize(3, num_selected);
Bx_near.resize(num_selected);
By_near.resize(num_selected);
Bz_near.resize(num_selected);

for (int i = 0; i < num_selected; ++i)
{
int index = selected_indices[i];
X_near.col(i) = X.col(index);
Bx_near(i) = Bx(index);
By_near(i) = By(index);
Bz_near(i) = Bz(index);
}
}

Vector3d generateRandomVelocity() {

    // Two points defining a parrallel and a perpendicular vector to the plane of launch (numbers taken from inventor model)
    Vector3d parr_vec_f = Vector3d(22.496172, 22.496172, 200.799134) * mm_to_m;
    Vector3d perp_vec_f = Vector3d(24.035402, 19.878480, 202.975935) * mm_to_m;

    Vector3d perp_vec = perp_vec_f - loding_pt;
    Vector3d parr_vec = parr_vec_f - loding_pt;

    // Normalize parallel direction
    Vector3d dir_parr = parr_vec.normalized();

    // Random number generators
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist_parr(50.0, 50.0); // Gaussian in the parallel direction
    normal_distribution<> dist_perp(0.0, 50.0); // Gaussian in the perpendicular direction
    uniform_real_distribution<> dist_angle(0.0, 2 * M_PI); // The perpindicular vector itself needs its direction randomized

    // Scale parallel component
    double v_parr = dist_parr(gen); // Speed, parallel
    Vector3d v_parallel = v_parr * dir_parr;

    // Get orthonormal basis for the plane perpendicular to dir_parr
    Vector3d u = perp_vec.normalized();
    Vector3d v = dir_parr.cross(u).normalized(); // Ensure u and v are orthogonal
    u = v.cross(dir_parr).normalized();

    // Random angle and speed for perpendicular component
    double theta = dist_angle(gen); // Direction, perpendicular
    double v_perp_mag = dist_perp(gen); // Speed, perpendicular

    Vector3d v_perpendicular = v_perp_mag * (cos(theta) * u + sin(theta) * v); 

    // Final velocity
    Vector3d total_velocity = v_parallel + v_perpendicular;

    return total_velocity;

}


PRECISION_TYPE calcBx(PRECISION_TYPE x) {




}

Vector3d diffEqnFunc(PRECISION_TYPE t, Vector3d r){
    // Diff eqn: d^2x/dt^2 = dv/dt = mu/m * grad(B) 

    PRECISION_TYPE dBx_dx = (calcBx(r(0) + half_grid_step) - calcBx(r(0) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBx_dy = (calcBy(r(1) + half_grid_step) - calcBy(r(1) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBx_dz = (calcBz(r(2) + half_grid_step) - calcBz(r(2) - half_grid_step)) / grid_step;
    
    PRECISION_TYPE dBy_dx = (calcBx(r(0) + half_grid_step) - calcBx(r(0) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBy_dy = (calcBy(r(1) + half_grid_step) - calcBy(r(1) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBy_dz = (calcBz(r(2) + half_grid_step) - calcBz(r(2) - half_grid_step)) / grid_step;

    PRECISION_TYPE dBz_dx = (calcBx(r(0) + half_grid_step) - calcBx(r(0) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBz_dy = (calcBy(r(1) + half_grid_step) - calcBy(r(1) - half_grid_step)) / grid_step;
    PRECISION_TYPE dBz_dz = (calcBz(r(2) + half_grid_step) - calcBz(r(2) - half_grid_step)) / grid_step;

    PRECISION_TYPE dv_x = (dBx_dx + dBy_dx + dBz_dx) * mu;
    PRECISION_TYPE dv_y = (dBx_dy + dBy_dy + dBz_dy) * mu;
    PRECISION_TYPE dv_z = (dBx_dz + dBy_dz + dBz_dz) * mu;

    Vector3d dv = Vector3d(dv_x, dv_y, dv_z);
    return dv;

}

Vector3d RK4Stepper(PRECISION_TYPE t, PRECISION_TYPE dt, Vector3d r, Vector3d v){

    Vector3d f1 = diffEqnFunc(t, r);
    Vector3d f2 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f1);
    Vector3d f3 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f2);
    Vector3d f4 = diffEqnFunc(t + dt, r + dt * f3);

    Vector3d dv = v + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4);

    return dv;
}

int main() {


    /*
    Problem Setup: Hydrogen Atom Trapping


    ! WILL ADD MORE PRECISION LATER 
    */
    
    // Time
    PRECISION_TYPE dt = 1e-9; // Time step in seconds
    PRECISION_TYPE t_0 = 0.0; // Initial time
    PRECISION_TYPE t_f = 1e-6; // Final time

    // Interpolation
    PRECISION_TYPE epsilon = 1.0; // Epsilon for RBF kernel


    const int num_particles = 1000;

    Vector3d initial_position = loding_pt; // Initial position of the particle
    for (int i = 0; i < num_particles; ++i) {
        Vector3d initial_velocity = generateRandomVelocity();
        // cout << "Velocity " << i + 1 << ": " << velocity.transpose() << endl;

        // Given an initial_velocity and our loading_pt, we advance using an RK4 stepper
        dv = RK4Stepper(t, dt, initial_position, initial_velocity);
        initial_position += dt * initial_velocity;
        initial_velocity = dv; // Update the velocity for the next step

    }




    return 0;
}