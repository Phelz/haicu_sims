/*
    Coding in a RK4 scheme for the Lorenz attractor.
*/
#include <Eigen/Dense>
#include <iostream>

using namespace std;

struct state
{
    Vector3d x; // Position
    Vector3d v; // Velocity
};


PRECISION_TYPE RKStepper(PRECISION_TYPE t,
                        PRECISION_TYPE dt,
                        state &state,
                        PRECISION_TYPE (*f)(PRECISION_TYPE, PRECISION_TYPE, state &))
{
    // RK4 stepper
    state k1 = f(t, dt, state);
    state k2 = f(t + 0.5 * dt, dt, {state.x + 0.5 * dt * k1.x, state.v + 0.5 * dt * k1.v});
    state k3 = f(t + 0.5 * dt, dt, {state.x + 0.5 * dt * k2.x, state.v + 0.5 * dt * k2.v});
    state k4 = f(t + dt, dt, {state.x + dt * k3.x, state.v + dt * k3.v});

    state.x += (dt / 6.0) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x);
    state.v += (dt / 6.0) * (k1.v + 2 * k2.v + 2 * k3.v + k4.v);

    return state;
}

PRECISION_TYPE LorenzAttractor(PRECISION_TYPE t,
                                PRECISION_TYPE dt,
                                state &state)
{
    // Lorenz attractor parameters
    double sigma = 10.0;
    double rho = 28.0;
    double beta = 8.0 / 3.0;

    // Lorenz equations
    state.v.x() = sigma * (state.x.y() - state.x.x());
    state.v.y() = state.x.x() * (rho - state.x.z()) - state.x.y();
    state.v.z() = state.x.x() * state.x.y() - beta * state.x.z();

    return state;
}
                            


int main()
{

    




}