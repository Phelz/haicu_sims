#include <vector>
#include <random>
#include <cmath>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

struct Vec3 {
    double x, y, z;

    Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }

    Vec3 cross(const Vec3& v) const {
        return {
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        };
    }

    double norm() const { return std::sqrt(x*x + y*y + z*z); }
};

// Magnetic field defined on a 3D grid
struct MagneticFieldGrid {
    std::vector<Vec3> B; // flattened grid
    int nx, ny, nz;
    double dx, dy, dz;
    Vec3 origin;

    // Trilinear interpolation of B field at position p
    Vec3 interpolate(const Vec3& p) const {
        double fx = (p.x - origin.x) / dx;
        double fy = (p.y - origin.y) / dy;
        double fz = (p.z - origin.z) / dz;

        int i = static_cast<int>(fx);
        int j = static_cast<int>(fy);
        int k = static_cast<int>(fz);

        if (i < 0 || j < 0 || k < 0 || i+1 >= nx || j+1 >= ny || k+1 >= nz)
            return {0, 0, 0}; // out of bounds

        double tx = fx - i;
        double ty = fy - j;
        double tz = fz - k;

        auto idx = [&](int x, int y, int z) { return x + nx * (y + ny * z); };

        Vec3 c000 = B[idx(i, j, k)];
        Vec3 c100 = B[idx(i+1, j, k)];
        Vec3 c010 = B[idx(i, j+1, k)];
        Vec3 c110 = B[idx(i+1, j+1, k)];
        Vec3 c001 = B[idx(i, j, k+1)];
        Vec3 c101 = B[idx(i+1, j, k+1)];
        Vec3 c011 = B[idx(i, j+1, k+1)];
        Vec3 c111 = B[idx(i+1, j+1, k+1)];

        Vec3 c00 = c000 * (1 - tx) + c100 * tx;
        Vec3 c10 = c010 * (1 - tx) + c110 * tx;
        Vec3 c01 = c001 * (1 - tx) + c101 * tx;
        Vec3 c11 = c011 * (1 - tx) + c111 * tx;

        Vec3 c0 = c00 * (1 - ty) + c10 * ty;
        Vec3 c1 = c01 * (1 - ty) + c11 * ty;

        return c0 * (1 - tz) + c1 * tz;
    }
};


void load_field_from_csv(const std::string& filename, MagneticFieldGrid& grid) {
    std::ifstream file(filename);
    std::string line;
    std::vector<double> xs, ys, zs;
    std::map<std::tuple<int, int, int>, Vec3> data;

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        double x, y, z, bx, by, bz;
        char comma;
        ss >> x >> comma >> y >> comma >> z >> comma >> bx >> comma >> by >> comma >> bz;
        xs.push_back(x);
        ys.push_back(y);
        zs.push_back(z);
        data[{(int)(x * 1e6), (int)(y * 1e6), (int)(z * 1e6)}] = {bx, by, bz};  // key for uniqueness
    }

    // Deduce grid size and spacing
    std::sort(xs.begin(), xs.end()); xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    std::sort(ys.begin(), ys.end()); ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
    std::sort(zs.begin(), zs.end()); zs.erase(std::unique(zs.begin(), zs.end()), zs.end());

    grid.nx = xs.size();
    grid.ny = ys.size();
    grid.nz = zs.size();
    grid.dx = xs[1] - xs[0];
    grid.dy = ys[1] - ys[0];
    grid.dz = zs[1] - zs[0];
    grid.origin = {xs[0], ys[0], zs[0]};
    grid.B.resize(grid.nx * grid.ny * grid.nz);

    auto idx = [&](int i, int j, int k) {
        return i + grid.nx * (j + grid.ny * k);
    };

    for (int i = 0; i < grid.nx; ++i)
        for (int j = 0; j < grid.ny; ++j)
            for (int k = 0; k < grid.nz; ++k)
                grid.B[idx(i, j, k)] = data[{(int)(xs[i] * 1e6), (int)(ys[j] * 1e6), (int)(zs[k] * 1e6)}];
}

Vec3 gradient_Bmag(const Vec3& pos, const MagneticFieldGrid& grid) {
    double delta = 1e-5;
    auto Bmag = [&](const Vec3& p) {
        Vec3 B = grid.interpolate(p);
        return std::sqrt(B.x*B.x + B.y*B.y + B.z*B.z);
    };

    double bx = (Bmag({pos.x + delta, pos.y, pos.z}) - Bmag({pos.x - delta, pos.y, pos.z})) / (2 * delta);
    double by = (Bmag({pos.x, pos.y + delta, pos.z}) - Bmag({pos.x, pos.y - delta, pos.z})) / (2 * delta);
    double bz = (Bmag({pos.x, pos.y, pos.z + delta}) - Bmag({pos.x, pos.y, pos.z - delta})) / (2 * delta);

    return {bx, by, bz};
}

// Sample velocity vector from 3D Gaussian
Vec3 sample_velocity(double mu, double sigma) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mu, sigma);
    return {dist(gen), dist(gen), dist(gen)};
}

void rk4_step(Vec3& pos, Vec3& vel, double dt, const MagneticFieldGrid& grid, double mu_over_m) {
    auto acceleration = [&](const Vec3& p) {
        Vec3 gradB = gradient_Bmag(p, grid);
        return gradB * mu_over_m;  // mu/m
    };

    Vec3 k1_v = acceleration(pos) * dt;
    Vec3 k1_x = vel * dt;

    Vec3 k2_v = acceleration(pos + k1_x * 0.5) * dt;
    Vec3 k2_x = (vel + k1_v * 0.5) * dt;

    Vec3 k3_v = acceleration(pos + k2_x * 0.5) * dt;
    Vec3 k3_x = (vel + k2_v * 0.5) * dt;

    Vec3 k4_v = acceleration(pos + k3_x) * dt;
    Vec3 k4_x = (vel + k3_v) * dt;

    vel = vel + (k1_v + k2_v * 2 + k3_v * 2 + k4_v) / 6;
    pos = pos + (k1_x + k2_x * 2 + k3_x * 2 + k4_x) / 6;
}
Vec3 compute_velocity_from_points(const Vec3& initial, const Vec3& final, double speed) {
    Vec3 direction = final - initial;
    double mag = std::sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    return direction * (speed / mag);
}

int main() {
    MagneticFieldGrid grid;
    load_field_from_csv("../data/bender/FA_Bield_Bender_Iso_100large_neg.csv", grid);

    Vec3 pos = {-0.0194, -0.021400000000000002, -0.1972270449692129};
    
    Vec3 initial_v = {-0.01293158, -0.02389436, -0.1880063};  // initial vition
    // Vec3 final_v = {-0.01177626, -0.008509994, -0.1783202};
    Vec3 final_v = {-0.006654615, -0.01198908, -0.179054};
    
    Vec3 vel = compute_velocity_from_points(initial_v, final_v, 30.0);  // 20 m/s

    // Vec3 vel = sample_velocity(100.0, 10.0); // m/s

    double dt = 1e-6;
    double mu = 9.27e-24;         // Bohr magneton
    double mass = 1.67e-27;       // proton mass
    double mu_over_m = mu / mass;

    std::ofstream outfile("../data/trajectory.csv");
    outfile << "x,y,z\n";

    for (int i = 0; i < 1000; ++i) {
        rk4_step(pos, vel, dt, grid, mu_over_m);
        outfile << pos.x << "," << pos.y << "," << pos.z << "\n";
    }

    outfile.close();
    return 0;
}

// int main() {
//     // Example magnetic field: uniform field in z
//     int nx = 10, ny = 10, nz = 10;
//     MagneticFieldGrid grid;
//     grid.nx = nx; grid.ny = ny; grid.nz = nz;
//     grid.dx = grid.dy = grid.dz = 0.01;
//     grid.origin = {0, 0, 0};
//     grid.B.resize(nx * ny * nz, {0, 0, 1}); // uniform B = 1 T in z

//     Vec3 pos = {0.045, 0.045, 0.045};
//     Vec3 vel = sample_velocity(100.0, 10.0); // m/s

//     double dt = 1e-6;
//     double charge = 1.6e-19;
//     double mass = 1.67e-27;

//     std::ofstream outfile("trajectory.csv");
//     outfile << "x,y,z\n";

//     for (int i = 0; i < 1000; ++i) {
//         rk4_step(pos, vel, dt, grid, charge, mass);
//         outfile << pos.x << "," << pos.y << "," << pos.z << "\n";
//     }

//     outfile.close();
//     return 0;
// }