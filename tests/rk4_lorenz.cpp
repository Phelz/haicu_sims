
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <constants.h>

using namespace std;
using namespace Eigen;

Vector3d diffEqnFunc(PRECISION_TYPE t, Vector3d r) {

    PRECISION_TYPE sigma = 10.0;
    PRECISION_TYPE beta = 8.0 / 3.0;
    PRECISION_TYPE rho = 28.0;

    Vector3d dr;
    dr(0) = sigma * (r(1) - r(0));
    dr(1) = r(0) * (rho - r(2)) - r(1);
    dr(2) = r(0) * r(1) - beta * r(2);

    return dr;
}


Vector3d RK4Stepper(PRECISION_TYPE t, PRECISION_TYPE dt, Vector3d r){

    Vector3d f1 = diffEqnFunc(t, r);
    cout << "f1: " << f1.transpose() << endl;
    cout << "r + (dt / 2) * f1: " << (r + (dt / 2) * f1).transpose() << endl;
    Vector3d f2 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f1);
    Vector3d f3 = diffEqnFunc(t + (dt / 2), r + (dt / 2) * f2);
    Vector3d f4 = diffEqnFunc(t + dt, r + dt * f3);

    Vector3d dr = r + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4);

    return dr;
}


int main() {

    
    PRECISION_TYPE t_0 = 0.0; // Initial time
    PRECISION_TYPE t_f = 10.0; // Final time
    PRECISION_TYPE dt = 0.01; // Time step

    Vector3d r_0;
    r_0(0) = -8.0; // Initial x position
    r_0(1) = 8.0; // Initial y position
    r_0(2) = 27.0; // Initial z position


    // Open a file to save the data
    ofstream file("../data/rk4_lorenz.csv");

    // Write the header to the file
    // Set the precision to 10 decimal places
    file.precision(10);
    file << "t,x,y,z" << endl;
    file << t_0 << "," << r_0(0) << "," << r_0(1) << "," << r_0(2) << endl;


    Vector3d dr;
    for (PRECISION_TYPE t = t_0; t <= t_f; t += dt) {
        // Calculate the new position and velocity
        dr = RK4Stepper(t, dt, r_0);
        
        // Write the data to the file
        
        // Update the position
        r_0 = dr;
        file << t << "," << r_0(0) << "," << r_0(1) << "," << r_0(2) << endl;

    }

    // close the file
    file.close();

    
    return 0; 
}