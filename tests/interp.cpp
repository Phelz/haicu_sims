#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <functional>
#include <vector>

#include <rbf_interpolator.hpp> // Ensure this includes GaussianRbfKernel
#include <constants.h>

using namespace std;
using namespace Eigen;
using namespace mathtoolbox;

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

int main()
{
    // Load the data
    vector<Vector3d> positions;
    vector<PRECISION_TYPE> Bx_vals, By_vals, Bz_vals;
    load_csv("../data/bender/FA_Bield_Bender_Iso_100large.csv", positions, Bx_vals, By_vals, Bz_vals);

    cout << "Constructing Matrices..." << endl;

    const int N = positions.size();
    MatrixXd X(3, N);
    for (int i = 0; i < N; ++i)
    {
        X.col(i) = positions[i];
    }
    MatrixXd X_transposed = X.transpose(); // 3 x N

    cout << "Constructed Position Matrix X" << endl;

    VectorXd Bx(N), By(N), Bz(N);
    for (int i = 0; i < N; ++i)
    {
        Bx(i) = Bx_vals[i];
        By(i) = By_vals[i];
        Bz(i) = Bz_vals[i];
    }
    cout << "Constructed Bx, By, Bz Vectors" << endl;

    // Interpolate at a query point
    Vector3d query_point(0.0, 0.0, -0.15);
    double r_tol = 0.005; // Radius tolerance
    double epsilon = 1.0;


    // Interpolators
    RbfInterpolator interp_Bx(GaussianRbfKernel(epsilon), true);
    RbfInterpolator interp_By(GaussianRbfKernel(epsilon), true);
    RbfInterpolator interp_Bz(GaussianRbfKernel(epsilon), true);

    cout << "Constructed Interpolators" << endl;

    // interp_Bx.SetData(X, Bx);
    // interp_By.SetData(X, By);
    // interp_Bz.SetData(X, Bz);

    MatrixXd X_near;
    VectorXd Bx_near, By_near, Bz_near;
    extract_points_within_radius(X, Bx, By, Bz, X_near, Bx_near, By_near, Bz_near, query_point, r_tol);

    interp_Bx.SetData(X_near, Bx_near);
    interp_By.SetData(X_near, By_near);
    interp_Bz.SetData(X_near, Bz_near);

    cout << "Successfully Set Data for Interpolators" << endl;

    interp_Bx.CalcWeights(true, 0.0);
    interp_By.CalcWeights(true, 0.0);
    interp_Bz.CalcWeights(true, 0.0);
    cout << "Successfully Calculated Weights for Interpolators" << endl;

    
    PRECISION_TYPE bx = interp_Bx.CalcValue(query_point);
    PRECISION_TYPE by = interp_By.CalcValue(query_point);
    PRECISION_TYPE bz = interp_Bz.CalcValue(query_point);

    cout << "Interpolated B at " << query_point.transpose() << ": ("
         << bx << ", " << by << ", " << bz << ")" << endl;

    return 0;

    // Particle parameters
    // Vector3d mu(0, 0, 1e-23);    // Magnetic moment
    // PRECISION_TYPE bohr_magneton = 9.27e-24; // Bohr magneton
    // PRECISION_TYPE mass = 1.67e-27;          // Mass of a hydrogen atom in kilograms
    // PRECISION_TYPE dt = 1e-9;
    // int num_steps = 1000;
    // Vector3d mu = Vector3d::Random().normalized() * bohr_magneton;

    // Initial state
    // particle.x = Vector3d(-0.0194, -0.021400000000000002, -0.1972270449692129); // Starting position

    // Vector3d v_initial(-0.01293158, -0.02389436, -0.1880063);
    // Vector3d v_final(-0.006654615, -0.01198908, -0.179054);
    // particle.v = v_final - v_initial; // Initial velocity


}

// int main()
// {
//     RbfInterpolator rbf_interpolator(
//         GaussianRbfKernel(epsilon), true);

//     return 0;
// }


// struct State
// {
//     Vector3d x; // Position
//     Vector3d v; // Velocity
// };

// class ParticleTracer
// {
// public:
//     ParticleTracer(const Vector3d &mu, double mass, double dt,
//                    const mathtoolbox::RbfInterpolator &interp_Bx,
//                    const mathtoolbox::RbfInterpolator &interp_By,
//                    const mathtoolbox::RbfInterpolator &interp_Bz,
//                    double epsilon)
//         : mu(mu), mass(mass), dt(dt),
//           interp_Bx(interp_Bx), interp_By(interp_By), interp_Bz(interp_Bz),
//           epsilon(epsilon) {}

//     void Step(State &state)
//     {
//         State k1 = Derivative(state);
//         State k2 = Derivative({state.x + 0.5 * dt * k1.x, state.v + 0.5 * dt * k1.v});
//         State k3 = Derivative({state.x + 0.5 * dt * k2.x, state.v + 0.5 * dt * k2.v});
//         State k4 = Derivative({state.x + dt * k3.x, state.v + dt * k3.v});

//         state.x += (dt / 6.0) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x);
//         state.v += (dt / 6.0) * (k1.v + 2 * k2.v + 2 * k3.v + k4.v);
//     }

// private:
//     Vector3d mu;
//     double mass;
//     double dt;
//     double epsilon;
//     const mathtoolbox::RbfInterpolator &interp_Bx;
//     const mathtoolbox::RbfInterpolator &interp_By;
//     const mathtoolbox::RbfInterpolator &interp_Bz;

//     State Derivative(const State &state)
//     {
//         Vector3d xq = state.x;

//         Vector3d grad_Bx = interp_Bx.CalcGradient(xq, epsilon);
//         Vector3d grad_By = interp_By.CalcGradient(xq, epsilon);
//         Vector3d grad_Bz = interp_Bz.CalcGradient(xq, epsilon);

//         Matrix3d J;
//         J.row(0) = grad_Bx.transpose();
//         J.row(1) = grad_By.transpose();
//         J.row(2) = grad_Bz.transpose();

//         Vector3d F = J.transpose() * mu;

//         return {state.v, F / mass};
//     }
// };
