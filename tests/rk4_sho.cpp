
#include <iostream>
#include <fstream>

using namespace std;

double diffEqnFunc(double t, double x) {
    // Define the differential equation for the simple harmonic oscillator
    double omega = 1.0; // Angular frequency
    return -omega * omega * x;
}

double RK4Stepper(double t, double dt, double x, double v){

    double f1 = diffEqnFunc(t, x);
    double f2 = diffEqnFunc(t + (dt / 2), x + (dt / 2) * f1);
    double f3 = diffEqnFunc(t + (dt / 2), x + (dt / 2) * f2);
    double f4 = diffEqnFunc(t + dt, x + dt * f3);

    double dv = v + (dt / 6) * (f1 + 2 * f2 + 2 * f3 + f4);
    return dv;
}


int main() {

    double omega_0 = 1.0; // Initial angular velocity
    double x_0 = 1.0; // Initial x position
    double v_0 = 0.0; // Initial velocity
    
    double t_0 = 0.0; // Initial time
    double t_f = 10.0; // Final time
    double dt = 0.01; // Time step


    // Open a file to save the data
    ofstream file("../data/rk4_sho.csv");

    // Write the header to the file
    file << "t,x,v" << endl;

    double dv = 0.0;

    for (double t = t_0; t <= t_f; t += dt) {
        // Calculate the new position and velocity
        dv = RK4Stepper(t, dt, x_0, v_0);
        x_0 += dt * v_0;
        v_0 = dv;        

        // Write the data to the file
        file << t << "," << x_0 << endl;


    }

    // close the file
    file.close();

    
    return 0; 
}