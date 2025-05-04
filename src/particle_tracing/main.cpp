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
#include <chrono>

#include <omp.h>

using namespace std;
using namespace Eigen;
using namespace mathtoolbox;



const PRECISION_TYPE mm_to_m = 1e-3;
const PRECISION_TYPE mass = 1.67e-27; // Mass of a hydrogen atom in kilograms
const PRECISION_TYPE bohr_magneton = 9.274e-24; // Bohr magneton in J/T
const PRECISION_TYPE mu = bohr_magneton/mass; // Magnetic moment per unit mass

const PRECISION_TYPE grid_step = 0.00295309; // Grid step in meters
const PRECISION_TYPE half_grid_step = grid_step / 2.0; // Half grid step

const PRECISION_TYPE r_tol = grid_step*2.5; // Radius tolerance
const PRECISION_TYPE epsilon = 1.0;


// The initial Loading point from which the atoms are launched in the simulation
const Vector3d loding_pt = Vector3d(-22.996172, -22.996172, -201.506241) * mm_to_m;



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
    Vector3d parr_vec_f = Vector3d(-22.496172, -22.496172, -200.799134) * mm_to_m;
    Vector3d perp_vec_f = Vector3d(-24.035402, -19.878480, -202.975935) * mm_to_m;

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


// PRECISION_TYPE calcBx(PRECISION_TYPE x) {




// }

Vector3d diffEqnFunc(PRECISION_TYPE t, Vector3d r, 
    const MatrixXd &X, const VectorXd &Bx, const VectorXd &By, const VectorXd &Bz,
    RbfInterpolator &interp_Bx, RbfInterpolator &interp_By, RbfInterpolator &interp_Bz) {
// Diff eqn: d^2x/dt^2 = dv/dt = mu/m * grad(B) 

    MatrixXd X_near;
    VectorXd Bx_near, By_near, Bz_near;
    extract_points_within_radius(X, Bx, By, Bz, X_near, Bx_near, By_near, Bz_near, r, r_tol);

    // If there's not enough points nearby, just return zero
    if (X_near.cols() < 3) {
        return Vector3d(0, 0, 0);
    }

    interp_Bx.SetData(X_near, Bx_near);
    interp_By.SetData(X_near, By_near);
    interp_Bz.SetData(X_near, Bz_near);

    interp_Bx.CalcWeights(true, 0.0);
    interp_By.CalcWeights(true, 0.0);
    interp_Bz.CalcWeights(true, 0.0);

    // Here, I use A Radial Basis Function (RBF) interpolator to get the B-field at various positions
    // The various positions in question correspond to a central finite difference scheme to calculate the gradient of the B-field

    Vector3d r_plus_dx = r + Vector3d(half_grid_step, 0, 0);
    Vector3d r_plus_dy = r + Vector3d(0, half_grid_step, 0);
    Vector3d r_plus_dz = r + Vector3d(0, 0, half_grid_step);

    Vector3d r_plus_2dx = r + Vector3d(2*half_grid_step, 0, 0);
    Vector3d r_plus_2dy = r + Vector3d(0, 2*half_grid_step, 0);
    Vector3d r_plus_2dz = r + Vector3d(0, 0, 2*half_grid_step);

    Vector3d r_minus_dx = r - Vector3d(half_grid_step, 0, 0);
    Vector3d r_minus_dy = r - Vector3d(0, half_grid_step, 0);
    Vector3d r_minus_dz = r - Vector3d(0, 0, half_grid_step);

    Vector3d r_minus_2dx = r - Vector3d(2*half_grid_step, 0, 0);
    Vector3d r_minus_2dy = r - Vector3d(0, 2*half_grid_step, 0);
    Vector3d r_minus_2dz = r - Vector3d(0, 0, 2*half_grid_step);
    
    // 4th order central difference
    PRECISION_TYPE dBx_dx = (-interp_Bx.CalcValue(r_plus_2dx) + 8*interp_Bx.CalcValue(r_plus_dx) - 8*interp_Bx.CalcValue(r_minus_dx) + interp_Bx.CalcValue(r_minus_2dx)) / (12*half_grid_step);
    PRECISION_TYPE dBx_dy = (-interp_By.CalcValue(r_plus_2dy) + 8*interp_By.CalcValue(r_plus_dy) - 8*interp_By.CalcValue(r_minus_dy) + interp_By.CalcValue(r_minus_2dy)) / (12*half_grid_step);
    PRECISION_TYPE dBx_dz = (-interp_Bz.CalcValue(r_plus_2dz) + 8*interp_Bz.CalcValue(r_plus_dz) - 8*interp_Bz.CalcValue(r_minus_dz) + interp_Bz.CalcValue(r_minus_2dz)) / (12*half_grid_step);

    PRECISION_TYPE dBy_dx = (-interp_Bx.CalcValue(r_plus_2dx) + 8*interp_Bx.CalcValue(r_plus_dx) - 8*interp_Bx.CalcValue(r_minus_dx) + interp_Bx.CalcValue(r_minus_2dx)) / (12*half_grid_step);
    PRECISION_TYPE dBy_dy = (-interp_By.CalcValue(r_plus_2dy) + 8*interp_By.CalcValue(r_plus_dy) - 8*interp_By.CalcValue(r_minus_dy) + interp_By.CalcValue(r_minus_2dy)) / (12*half_grid_step);
    PRECISION_TYPE dBy_dz = (-interp_Bz.CalcValue(r_plus_2dz) + 8*interp_Bz.CalcValue(r_plus_dz) - 8*interp_Bz.CalcValue(r_minus_dz) + interp_Bz.CalcValue(r_minus_2dz)) / (12*half_grid_step);

    PRECISION_TYPE dBz_dx = (-interp_Bx.CalcValue(r_plus_2dx) + 8*interp_Bx.CalcValue(r_plus_dx) - 8*interp_Bx.CalcValue(r_minus_dx) + interp_Bx.CalcValue(r_minus_2dx)) / (12*half_grid_step);
    PRECISION_TYPE dBz_dy = (-interp_By.CalcValue(r_plus_2dy) + 8*interp_By.CalcValue(r_plus_dy) - 8*interp_By.CalcValue(r_minus_dy) + interp_By.CalcValue(r_minus_2dy)) / (12*half_grid_step);
    PRECISION_TYPE dBz_dz = (-interp_Bz.CalcValue(r_plus_2dz) + 8*interp_Bz.CalcValue(r_plus_dz) - 8*interp_Bz.CalcValue(r_minus_dz) + interp_Bz.CalcValue(r_minus_2dz)) / (12*half_grid_step);


    // Instead of construct the Jacobian and do matrix multiplication, this is easier to read
    PRECISION_TYPE dv_x = (dBx_dx + dBy_dx + dBz_dx) * mu;
    PRECISION_TYPE dv_y = (dBx_dy + dBy_dy + dBz_dy) * mu;
    PRECISION_TYPE dv_z = (dBx_dz + dBy_dz + dBz_dz) * mu;

    Vector3d dv = Vector3d(dv_x, dv_y, dv_z);
    return dv;

}

// Vector3d diffEqnFunc(PRECISION_TYPE t, Vector3d r, 
//                     const MatrixXd &X, const VectorXd &Bx, const VectorXd &By, const VectorXd &Bz,
//                     RbfInterpolator &interp_Bx, RbfInterpolator &interp_By, RbfInterpolator &interp_Bz) {
//     // Diff eqn: d^2x/dt^2 = dv/dt = mu/m * grad(B) 

//     MatrixXd X_near;
//     VectorXd Bx_near, By_near, Bz_near;
//     extract_points_within_radius(X, Bx, By, Bz, X_near, Bx_near, By_near, Bz_near, r, r_tol);

//     // If there's not enough points nearby, just return zero
//     if (X_near.cols() < 3) {
//         return Vector3d(0, 0, 0);
//     }
    
//     interp_Bx.SetData(X_near, Bx_near);
//     interp_By.SetData(X_near, By_near);
//     interp_Bz.SetData(X_near, Bz_near);

//     interp_Bx.CalcWeights(true, 0.0);
//     interp_By.CalcWeights(true, 0.0);
//     interp_Bz.CalcWeights(true, 0.0);

//     // Here, I use A Radial Basis Function (RBF) interpolator to get the B-field at various positions
//     // The various positions in question correspond to a central finite difference scheme to calculate the gradient of the B-field

//     Vector3d r_plus_dx = r + Vector3d(half_grid_step, 0, 0);
//     Vector3d r_plus_dy = r + Vector3d(0, half_grid_step, 0);
//     Vector3d r_plus_dz = r + Vector3d(0, 0, half_grid_step);

//     Vector3d r_minus_dx = r - Vector3d(half_grid_step, 0, 0);
//     Vector3d r_minus_dy = r - Vector3d(0, half_grid_step, 0);
//     Vector3d r_minus_dz = r - Vector3d(0, 0, half_grid_step);



//     PRECISION_TYPE dBx_dx = (interp_Bx.CalcValue(r_plus_dx) - interp_Bx.CalcValue(r_minus_dx)) / grid_step;
//     PRECISION_TYPE dBx_dy = (interp_By.CalcValue(r_plus_dy) - interp_By.CalcValue(r_minus_dy)) / grid_step;
//     PRECISION_TYPE dBx_dz = (interp_Bz.CalcValue(r_plus_dz) - interp_Bz.CalcValue(r_minus_dz)) / grid_step;


//     PRECISION_TYPE dBy_dx = (interp_Bx.CalcValue(r_plus_dx) - interp_Bx.CalcValue(r_minus_dx)) / grid_step;
//     PRECISION_TYPE dBy_dy = (interp_By.CalcValue(r_plus_dy) - interp_By.CalcValue(r_minus_dy)) / grid_step;
//     PRECISION_TYPE dBy_dz = (interp_Bz.CalcValue(r_plus_dz) - interp_Bz.CalcValue(r_minus_dz)) / grid_step;

//     PRECISION_TYPE dBz_dx = (interp_Bx.CalcValue(r_plus_dx) - interp_Bx.CalcValue(r_minus_dx)) / grid_step;
//     PRECISION_TYPE dBz_dy = (interp_By.CalcValue(r_plus_dy) - interp_By.CalcValue(r_minus_dy)) / grid_step;
//     PRECISION_TYPE dBz_dz = (interp_Bz.CalcValue(r_plus_dz) - interp_Bz.CalcValue(r_minus_dz)) / grid_step;

//     // Instead of construct the Jacobian and do matrix multiplication, this is easier to read
//     PRECISION_TYPE dv_x = (dBx_dx + dBy_dx + dBz_dx) * mu;
//     PRECISION_TYPE dv_y = (dBx_dy + dBy_dy + dBz_dy) * mu;
//     PRECISION_TYPE dv_z = (dBx_dz + dBy_dz + dBz_dz) * mu;

//     Vector3d dv = Vector3d(dv_x, dv_y, dv_z);
//     return dv;

// }

Vector3d RK4Stepper(PRECISION_TYPE t, PRECISION_TYPE dt, Vector3d r, Vector3d v, 
                    const MatrixXd &X, const VectorXd &Bx, const VectorXd &By, const VectorXd &Bz,
                    RbfInterpolator &interp_Bx, RbfInterpolator &interp_By, RbfInterpolator &interp_Bz) {

    Vector3d f1 = diffEqnFunc(t, r, X, Bx, By, Bz, interp_Bx, interp_By, interp_Bz);
    Vector3d f2 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f1, X, Bx, By, Bz, interp_Bx, interp_By, interp_Bz);
    Vector3d f3 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f2, X, Bx, By, Bz, interp_Bx, interp_By, interp_Bz);
    Vector3d f4 = diffEqnFunc(t + dt, r + dt * f3, X, Bx, By, Bz, interp_Bx, interp_By, interp_Bz);

    Vector3d dv = v + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4);

    return dv;
}

int main() {


    /*
    Problem Setup: Hydrogen Atom Trapping


    ! WILL ADD MORE PRECISION LATER 
    */
    
    // Time
    PRECISION_TYPE dt = 2e-6; // Time step in seconds
    PRECISION_TYPE t_0 = 0.0; // Initial time
    PRECISION_TYPE t_f = 1e-3; // Final time

    // Interpolators
    RbfInterpolator interp_Bx(GaussianRbfKernel(epsilon), true);
    RbfInterpolator interp_By(GaussianRbfKernel(epsilon), true);
    RbfInterpolator interp_Bz(GaussianRbfKernel(epsilon), true);


    // Load the data
    vector<Vector3d> positions;
    vector<PRECISION_TYPE> Bx_vals, By_vals, Bz_vals;
    load_csv("../data/bfield/FA_Bfield_Total_Iso_100large.csv", positions, Bx_vals, By_vals, Bz_vals);


    cout << "Constructing Matrices..." << endl;

    const int N = positions.size();
    MatrixXd X(3, N);
    for (int i = 0; i < N; ++i)
    {
        X.col(i) = positions[i];
    }
    MatrixXd X_transposed = X.transpose(); // 3 x N


    VectorXd Bx(N), By(N), Bz(N);
    for (int i = 0; i < N; ++i)
    {
        Bx(i) = Bx_vals[i];
        By(i) = By_vals[i];
        Bz(i) = Bz_vals[i];
    }

    // Parrallel processing
    omp_set_dynamic(0);  // Disable dynamic adjustment of threads
    omp_set_num_threads(24); // Set the number of threads to use

    const int num_particles = 100;
    Vector3d dv;

    // Parallelize the particle tracing
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_particles; ++i) {
        Vector3d initial_position = loding_pt; // Initial position of the particle
        Vector3d initial_velocity = generateRandomVelocity();

        // Open a file to store the output for this particle
        ofstream output_file("../data/traces/particle_" + to_string(i + 1) + "_trajectory.csv");
        output_file.precision(16); 
        if (!output_file.is_open()) {
            cerr << "Failed to open output file for particle " << i + 1 << endl;
            exit(1);
        }

        // Write the header
        output_file << "t,x,y,z" << endl;

        // Start timing for this particle
        auto start_time = chrono::high_resolution_clock::now();

        // Copy the matrices and interpolators for this thread
        MatrixXd X_copy = X;
        VectorXd Bx_copy = Bx, By_copy = By, Bz_copy = Bz;
        RbfInterpolator interp_Bx_copy = interp_Bx, interp_By_copy = interp_By, interp_Bz_copy = interp_Bz;

        for (PRECISION_TYPE t = t_0; t < t_f; t += dt) {
            // Given an initial_velocity and our loading_pt, we advance using an RK4 stepper



            dv = RK4Stepper(t, dt, initial_position, initial_velocity, X_copy, Bx_copy, By_copy, Bz_copy, interp_Bx_copy, interp_By_copy, interp_Bz_copy);

            initial_position += dt * initial_velocity;
            initial_velocity = dv; // Update the velocity for the next step

            // if ever the particle goes out of  the grid bounds, terminate
            if (initial_position.x() < FINAL_XMIN || initial_position.x() > FINAL_XMAX ||
                initial_position.y() < FINAL_YMIN || initial_position.y() > FINAL_YMAX ||
                initial_position.z() < FINAL_ZMIN || initial_position.z() > FINAL_ZMAX) {
                break;
            }

            // Write the current time and position to the output file
            output_file << t << "," << initial_position.x() << "," << initial_position.y() << "," << initial_position.z() << endl;
        }

        // End timing for this particle
        auto end_time = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
        cout << "Particle " << i + 1 << " simulation time: " << duration << " s" << endl;

        output_file.close();
    }


    return 0;
}