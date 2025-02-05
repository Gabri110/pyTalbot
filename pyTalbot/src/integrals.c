/*
C code to compute the integrals.

Credits to Jérôme Richard, who explained in this stackoverflow's answer stackoverflow.com/a/79360271/24208929 
how to efficiently evaluate integrals in Python through the GSL library.
*/

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>

// We multiply the integrand by 100 and then divide the end result by 1000 to mitigate roundoff errors
double integrand_sin(double tau, void* generic_params)
{
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(fmax(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.

    if(u < epsilon)
        return sin(omega * tau) * k * 0.5 * 1000.; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return sin(omega * tau) * gsl_sf_bessel_J1(k * u) / u * 1000.;
}

// We multiply the integrand by 100 and then divide the end result by 1000 to mitigate roundoff errors
double integrand_cos(double tau, void* generic_params)
{
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(fmax(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.
    
    if(u < epsilon)
        return cos(omega * tau) * k * 0.5 * 1000.; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return cos(omega * tau) * gsl_sf_bessel_J1(k * u) / u * 1000.;
}


void compute_integrals(double* partial_integral_cos, double* partial_integral_sin, double* x_min, double* x_max, 
                        double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, 
                        int limit, double epsabs, double epsrel, 
                        double omega, int start, int end, double epsilon)
{

    for (int n = fmax(start,1); n < end; ++n) // We skip the n,t,z=0 cases as they are trivially null.
    {

        #pragma omp parallel for schedule(dynamic)
        for (int t = 1; t < t_size; ++t)
        {
            FILE *stream;

            // Print the iteration in clog.txt
            if (t == 1){
                stream = fopen("clog.txt", "a");
                fprintf(stream, "The value of n is: %d. \n",n);
                fflush(stream);
                fclose (stream);
            }
            
            gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1024*1024*8);
            gsl_integration_cquad_workspace* workspace_cquad = gsl_integration_cquad_workspace_alloc(1024*1024*2);

            double err __attribute__((unused));
            

            double params[5];

            gsl_function func_cos;
            func_cos.function = &integrand_cos;
            func_cos.params = &params;

            gsl_function func_sin;
            func_sin.function = &integrand_sin;
            func_sin.params = &params;

            params[0] = k_n_values[n];
            params[1] = t_values[t];
            params[3] = omega;
            params[4] = epsilon;

            int status;

            gsl_set_error_handler_off();

            for (int z = 1; z < z_size; ++z)
            {
                params[2] = z_values[z];

                status = gsl_integration_qag(&func_cos, x_min[t*z_size+z], x_max[t*z_size+z], 
                             epsabs, epsrel, limit, 5, workspace, 
                             &partial_integral_cos[((n-start) * t_size+t)*z_size+z], &err);
                    
                if (status) {
                                status = gsl_integration_cquad(&func_cos, x_min[t*z_size+z], x_max[t*z_size+z], 
                                    epsabs, epsrel, workspace_cquad, 
                                    &partial_integral_cos[((n-start) * t_size+t)*z_size+z], NULL, NULL);

                                if (status) {
                                    stream = fopen("clog.txt", "a");
                                    fprintf(stream, "Error at cosine for n = %d, z = %d and t = %d \n",n,z,t);
                                    fflush(stream);
                                    fclose (stream);

                                    partial_integral_cos[((n-start) * t_size+t)*z_size+z] = 0.;
                                }
                            }
                    
                status = gsl_integration_qag(&func_sin, x_min[t*z_size+z], x_max[t*z_size+z], 
                             epsabs, epsrel, limit, 5, workspace, 
                             &partial_integral_sin[((n-start) * t_size+t)*z_size+z], &err);
                    
                if (status) {
                                status = gsl_integration_cquad(&func_sin, x_min[t*z_size+z], x_max[t*z_size+z], 
                                    epsabs, epsrel, workspace_cquad, 
                                    &partial_integral_cos[((n-start) * t_size+t)*z_size+z], NULL, NULL);

                                if (status) {
                                    stream = fopen("clog.txt", "a");
                                    fprintf(stream, "Error at sine for n = %d, z = %d and t = %d \n",n,z,t);
                                    fflush(stream);
                                    fclose (stream);

                                    partial_integral_sin[((n-start) * t_size+t)*z_size+z] = 0.;
                                }
                            }


                // We multiply the integrand by 100 and then divide the end result by 1000 to mitigate roundoff errors
                partial_integral_cos[((n-start) * t_size+t)*z_size+z] /= 1000.;
                partial_integral_sin[((n-start) * t_size+t)*z_size+z] /= 1000.;

            }

            // We free the memory
            gsl_integration_workspace_free(workspace);
            gsl_integration_cquad_workspace_free(workspace_cquad);
        }
    }
}