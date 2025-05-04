#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
// #include "lib.h"

using namespace std;

// output file as global variable
ofstream ofile;

// function declarations
void derivatives(double, double*, double*);
void initialise(double&, double&, int&);
void output(double, double*, double);
void runge_kutta_4(double*, double*, int, double, double, double*, void (*)(double, double*, double*));

int main(int argc, char* argv[])
{
    // declarations of variables
    double *y, *dydt, *yout, t, h, tmax, E0;
    double initial_x = 1.0; // predefined initial position
    double initial_v = 0.0; // predefined initial velocity
    int i, number_of_steps = 1000; // predefined number of steps
    int n;
    const char* outfilename = "output.txt"; // predefined output file name

    // // Read in output file, abort if there are too few command-line arguments
    // if (argc <= 1)
    // {
    //     cout << "Bad Usage: " << argv[0] << " read also output file on same line" << endl;
    //     exit(1);
    // }
    // else
    // {
    //     outfilename = argv[1];
    // }

    ofile.open(outfilename);

    // number of differential equations
    n = 2;

    // allocate memory for the arrays
    dydt = new double[n];
    y = new double[n];
    yout = new double[n];

    // read in the initial position, velocity and number of steps
    initialise(initial_x, initial_v, number_of_steps);

    // setting initial values, step size and max time
    h = 4. * acos(-1.) / ((double)number_of_steps);
    tmax = h * number_of_steps;

    y[0] = initial_x;   // initial position
    y[1] = initial_v;   // initial velocity
    t = 0.;             // initial time

    E0 = 0.5 * y[0] * y[0] + 0.5 * y[1] * y[1]; // initial total energy

    // start solving the differential equations using the RK4 method
    while (t <= tmax)
    {
        derivatives(t, y, dydt);
        runge_kutta_4(y, dydt, n, t, h, yout, derivatives);

        for (i = 0; i < n; i++)
        {
            y[i] = yout[i];
        }

        t += h;
        output(t, y, E0);
    }

    delete[] y;
    delete[] dydt;
    delete[] yout;

    ofile.close();
    return 0;
}

// Read in from screen the number of steps, initial position and initial speed
void initialise(double& initial_x, double& initial_v, int& number_of_steps)
{
    cout << "Initial position = ";
    cin >> initial_x;
    cout << "Initial speed = ";
    cin >> initial_v;
    cout << "Number of steps = ";
    cin >> number_of_steps;
}

// Sets up the derivatives for this special case
void derivatives(double t, double* y, double* dydt)
{
    dydt[0] = y[1];     // derivative of x
    dydt[1] = -y[0];    // derivative of v
}

// Write out the final results
void output(double t, double* y, double E0)
{
    ofile << setiosflags(ios::showpoint | ios::uppercase);
    ofile << setw(15) << setprecision(8) << t;
    ofile << setw(15) << setprecision(8) << y[0];
    ofile << setw(15) << setprecision(8) << y[1];
    ofile << setw(15) << setprecision(8) << cos(t);
    ofile << setw(15) << setprecision(8) << 0.5 * y[0] * y[0] + 0.5 * y[1] * y[1] - E0 << endl;
}

// 4th order Runge-Kutta method
void runge_kutta_4(double* y, double* dydx, int n, double x, double h, double* yout, void (*derivs)(double, double*, double*))
{
    int i;
    double xh, hh, h6;
    double* dym = new double[n];
    double* dyt = new double[n];
    double* yt = new double[n];

    hh = h * 0.5;
    h6 = h / 6.;
    xh = x + hh;

    for (i = 0; i < n; i++)
    {
        yt[i] = y[i] + hh * dydx[i];
    }
    (*derivs)(xh, yt, dyt);

    for (i = 0; i < n; i++)
    {
        yt[i] = y[i] + hh * dyt[i];
    }
    (*derivs)(xh, yt, dym);

    for (i = 0; i < n; i++)
    {
        yt[i] = y[i] + h * dym[i];
        dym[i] += dyt[i];
    }
    (*derivs)(x + h, yt, dyt);

    for (i = 0; i < n; i++)
    {
        yout[i] = y[i] + h6 * (dydx[i] + dyt[i] + 2.0 * dym[i]);
    }

    delete[] dym;
    delete[] dyt;
    delete[] yt;
}
