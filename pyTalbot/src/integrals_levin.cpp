/*
C++ code to compute the integrals using the Levin method.

Credits to ........
*/

#include <bestlime/Bessel_integrator.hpp>
#include <bestlime/config.hpp>
#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>


// class implementing the variable transform u(z) = √(z^2 + r^2)
// The allowed limits for z are 0 <= z < `infinity`.
class Sqrt_r_grid : public bestlime::Grid_1d
{
    public:
        Sqrt_r_grid(const bestlime::index_vector& n_points,
                const bestlime::vector_d& z_limits,
                const bestlime::vector_d& par)
        : bestlime::Grid_1d(n_points, z_limits, par)
        {
        if (par.size() != 1)
            throw std::logic_error("Sqrt_r_grid: wrong number of parameters.");

        _r = par.at(0);
        _initialize_base_grid(n_points, z_limits);
        }

        // transform u(z)
        double variable_transform(double z) const override
        {
        return std::sqrt(z*z + _r*_r);
        }

        // inverse transform z(u)
        double inverse_transform(double u) const override
        {
        return std::sqrt(u*u - _r*_r);
        }

        // Jacobian du/dz as a function of u.
        double jacobian(double u) const override
        {
            double a = _r/u;
        return std::sqrt(1. - a*a);
        }

        // Hessian d^2u/dz^2 as a function of u.
        double hessian(double u) const override
        {
        return _r*_r / u*u*u;
        }

    private:
        double _r;
};


std::function<double(double, double, double)>
    integrand_sin= [](double u, double z, double omega)
{
    const double r = std::sqrt(u*u + z*z); // Precompute this for efficiency.

    return std::sin(omega * r) / r;
};

std::function<double(double, double, double)>
    integrand_cos= [](double u, double z, double omega)
{
    const double r = std::sqrt(u*u + z*z); // Precompute this for efficiency.

    return std::cos(omega * r) / r;
};


extern "C" void perform_integrals(double* partial_integral, double* x_min, double* x_max, 
                        double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, 
                        double omega, bool sin, int points=50)
{
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int z = 1; z < z_size; ++z) // We skip the n,t,z=0 cases as they are trivially null.
    {
        std::cout << "Iteration: " << z << "/" << z_size << std::endl;
        // quantities defining the interpolation grid
        const bestlime::index_vector n_points {points};

        for (int t = 1; t < t_size; ++t)
        {
            // We make sure that x_max > x_min
            if (x_min[t*z_size+z] < x_max[t*z_size+z]){
                // quantities defining the interpolation grid
                const bestlime::vector_d integration_limits { x_min[t*z_size+z], x_max[t*z_size+z]};

                // Bessel integrator with custom grid and nu=1
                bestlime::Bessel_integrator<Sqrt_r_grid> bessel_int_sqrt_r(n_points, integration_limits, {z_values[z]});

                // discretize integrand on the grid
                // note: template argument of of discretize() is deduced
                 bestlime::grid_vector f_values;

                if (sin) {
                    f_values = {bessel_int_sqrt_r.discretize(integrand_sin, z_values[z], omega)};
                } else {
                    f_values = {bessel_int_sqrt_r.discretize(integrand_cos, z_values[z], omega)};
                }
                // THE PROBLEM IS AFTER THIS

                // We perform the integrals
                for (int n = 1; n < n_size; ++n) 
                {
                    partial_integral[(n*t_size+t)*z_size+z] = bessel_int_sqrt_r.int_J_nu(f_values, k_n_values[n]);
                }
            }
        }
    }
}