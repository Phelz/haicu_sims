/*
    Testing the RK4 integrator for a simple harmonic oscillator in 1D.
*/

#include <constants.h>

PRECISION_TYPE calcEnergy(PRECISION_TYPE v,
                          PRECISION_TYPE x,
                          PRECISION_TYPE k,
                          PRECISION_TYPE m)
{
    return 0.5 * k * x * x + 0.5 * v * v;
}

int main()
{

    // Define the block's parameters
    PRECISION_TYPE v_0 = 0.0;     // initial velocity
    PRECISION_TYPE x_0 = 4.0;     // (equilibrium) initial position
    PRECISION_TYPE omega_0 = 1.0; // angular frequency = sqrt(k/m)
    PRECISION_TYPE m = 1.0;       // mass

    // Time parameters
    int num_timesteps = 1000;
    PRECISION_TYPE t_f = 2 * PI;
    PRECISION_TYPE t_i = 0;
    PRECISION_TYPE dt = (t_f - t_i) / num_timesteps;

    // Initial Energy:

    return 0;
}