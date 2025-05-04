#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include <constants.h>

using namespace std;
using namespace Eigen;

const PRECISION_TYPE mm_to_m = 1e-3;

Vector3d generateRandomVelocity() {

    Vector3d loding_pt = Vector3d(22.996172, 22.996172, 201.506241) * mm_to_m;
    Vector3d parr_vec_f = Vector3d(22.496172, 22.496172, 200.799134) * mm_to_m;
    Vector3d perp_vec_f = Vector3d(24.035402, 19.878480, 202.975935) * mm_to_m;

    Vector3d perp_vec = perp_vec_f - loding_pt;
    Vector3d parr_vec = parr_vec_f - loding_pt;


    // Normalize parallel direction
    Vector3d dir_parr = parr_vec.normalized();

    // Random number generators
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> dist_parr(50.0, 50.0);
    normal_distribution<> dist_perp(0.0, 50.0);
    uniform_real_distribution<> dist_angle(0.0, 2 * M_PI);

    // Scale parallel component
    double v_parr = dist_parr(gen);
    Vector3d v_parallel = v_parr * dir_parr;

    // Get orthonormal basis for the plane perpendicular to dir_parr
    Vector3d u = perp_vec.normalized();
    Vector3d v = dir_parr.cross(u).normalized(); // Ensure u and v are orthogonal

    // Re-orthogonalize u (in case not perfectly perpendicular)
    u = v.cross(dir_parr).normalized();

    // Random angle and speed for perpendicular component
    double theta = dist_angle(gen);
    double v_perp_mag = dist_perp(gen);

    Vector3d v_perpendicular = v_perp_mag * (cos(theta) * u + sin(theta) * v);

    // Final velocity
    Vector3d total_velocity = v_parallel + v_perpendicular;

    return total_velocity;

}


int main() {

    for (int i = 0; i < 1000; ++i) {
        Vector3d velocity = generateRandomVelocity();
        cout << "Velocity " << i + 1 << ": " << velocity.transpose() << endl;
    }

    return 0;

}

