/*
cos_integral.cpp
Compile with    g++ -O3 cos_integral.cpp -lgsl -lgslcblas --shared -fPIC -o lib_cos_integral.so -fopenmp

Credits to Jérôme Richard, who explained in this stackoverflow's answer stackoverflow.com/a/79360271/24208929 
how to efficiently evaluate integrals in Python through the GSL library.
*/

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>

double integrand_sin(double tau, void* generic_params)
{
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(std::max(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.

    if(u < epsilon)
        return sin(omega * tau) * k * 0.5; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return sin(omega * tau) * gsl_sf_bessel_J1(k * u) / u;
}

double integrand_cos(double tau, void* generic_params)
{
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(std::max(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.
    
    if(u < epsilon)
        return cos(omega * tau) * k * 0.5; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return cos(omega * tau) * gsl_sf_bessel_J1(k * u) / u;
}


extern "C" void compute_cos(double* partial_integral, double* x_min, double* x_max, 
                        double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, 
                        int limit, double epsabs, double epsrel, 
                        double omega, double epsilon)
{
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 1; n < n_size; ++n) // We skip the n,t,z=0 cases as they are trivially null.
    {
        for (int t = 1; t < t_size; ++t)
        {
            gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1024*1024*1024);

            double params[5];
            double err [[maybe_unused]];

            gsl_function func_cos;
            func_cos.function = &integrand_cos;
            func_cos.params = &params;

            for (int z = 1; z < z_size; ++z)
            {
                params[0] = k_n_values[n];
                params[1] = t_values[t];
                params[2] = z_values[z];
                params[3] = omega;
                params[4] = epsilon;

                gsl_integration_qags(&func_cos, x_min[t*z_size+z], x_max[t*z_size+z], 
                                     epsabs, epsrel, limit, workspace, 
                                     &partial_integral[(n*t_size+t)*z_size+z], &err);
            }

            gsl_integration_workspace_free(workspace);
        }
    }
}


extern "C" void compute_sin(double* partial_integral, double* x_min, double* x_max, 
                        double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, 
                        int limit, double epsabs, double epsrel, 
                        double omega, double epsilon)
{
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 1; n < n_size; ++n) // We skip the n,t,z=0 cases as they are trivially null.
    {
        for (int t = 1; t < t_size; ++t)
        {
            gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1024*1024*1024);

            double params[5];
            double err [[maybe_unused]];

            gsl_function func_sin;
            func_sin.function = &integrand_sin;
            func_sin.params = &params;

            for (int z = 1; z < z_size; ++z)
            {
                params[0] = k_n_values[n];
                params[1] = t_values[t];
                params[2] = z_values[z];
                params[3] = omega;
                params[4] = epsilon;

                gsl_integration_qags(&func_sin, x_min[t*z_size+z], x_max[t*z_size+z], 
                                     epsabs, epsrel, limit, workspace, 
                                     &partial_integral[(n*t_size+t)*z_size+z], &err);
            }

            gsl_integration_workspace_free(workspace);
        }
    }
}