/* Reproducing numerical potential for figure 3.5, subfigure 3 on page 53 for Clement Baruteau's PhD-thesis. For large input arrays it is many times faster than the Python code. */

#include<stdio.h>
#include<math.h>									       
#include<complex.h> // This library is declared before fftw3.h. 
#include<fftw3.h> // Import the fftw3 header

#define PI 3.1415926535
#define NPOINTS 100000 // Number of sample points
#define ESP 0.01 // Softening parameter. Specified in the text, page 52
#define SHIFT 0.1 // Shifting the peak of the linear density to x = 0.1
#define X_MIN 0.0 // Minimum value of x
#define X_MAX 1.0 // Maximum value of x
#define STEP (X_MAX-X_MIN)/NPOINTS // Uniform stepsize/spacing in x
#define G 1.0 // Gravitational constant in code units

double Kernel(double x, double esp, int n, int i){ // Equation 3.32.b, page 52 to calculate kernel
  pow(pow(x,2)+pow(esp,2),-0.5);
}

// A Gaussian is used as an approximation to the singular dirac delta function
double Gauss(double x, double shift, int n){ 
  double sigma = 1./n; // The Gaussian is made very narrow (standard deviation)
  return 1./(sqrt(2*PI)*sigma)*exp(-pow(x-shift,2)/(2*pow(sigma,2)));
}

// Define analytical potential
double V_A_func(double x, double shift, double esp){
  return -G/sqrt(pow(x-shift,2)+pow(esp,2));   
}

int main(void){

  int i; // Declare the loop counter
  double *in; // Declaring a pointer for the input (real array)
  fftw_complex *out; // Declaring a pointer for the output (complex array)

  fftw_complex *K_out; // Declaring a pointer for the Kernel output array
  fftw_complex *ld_out; // Declaring a pointer for the linear density output array
    
  fftw_plan plan; // Declaring a plan of variable type "fftw_plan"

  double x[NPOINTS]; // Declare and initialize a static array for x (the sample points) 
  for (int i = 0; i < NPOINTS; i++){
    x[i] = i*(X_MAX-X_MIN)/(NPOINTS-1); // uniform spacing between each x (includes each endpoint)
  }

  double V_A[NPOINTS]; // Calculate the analytical potential for each x-value
  for (int i = 0; i < NPOINTS; i++){
    V_A[i] = V_A_func(x[i], SHIFT, ESP);
  }

#ifdef PRINT // Print x-array if condition is specified when compiling
  for (int i = 0; i < NPOINTS; i++){
    printf("x-array: %d %f \n", i, x[i]);
  }
#endif

  // Allocating memory for input and output array for both the input and output array
  in = (double*) fftw_malloc(sizeof(double)*NPOINTS);	
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NPOINTS);
  ld_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NPOINTS); // Needs its own memory

  plan = fftw_plan_dft_r2c_1d(NPOINTS, in, out, FFTW_ESTIMATE); // Initializing the plan

  // Initializing linear density input array:
  printf("\n Coefficients of the linear density array:\n\n");
  for(i = 0; i < NPOINTS; i++){  
    in[i] = Gauss(x[i], SHIFT, NPOINTS); // Use the input function
#ifdef PRINT
    printf("ld-array: %d %f \n", i, in[i]);
#endif
  }
  printf("\n");

  fftw_execute(plan); // Execution of FFT on the plan (making the actual DFT of the linear density)

  for(i = 0; i < (NPOINTS/2+1); i++){
    ld_out[i] = out[i]; // Saving the output into an array for linear density transforms 
#ifdef PRINT
    printf("%d %11.7f %11.7f \n", i, creal(ld_out[i]), cimag(ld_out[i]));
#endif
  }

  // Initializing kernel input array
  printf("\n Coefficients of the kernel array:\n\n");
  for(i = 0; i < NPOINTS; i++){  
    if (i < NPOINTS/2){
      in[i] = Kernel(x[i], ESP, NPOINTS, i);
    }
    else{
      in[i] = Kernel(x[NPOINTS-i], ESP, NPOINTS, i);
    }  
#ifdef PRINT
    printf("K-array: %d %f \n", i, in[i]);
#endif
  }
  printf("\n");

  fftw_execute(plan); // Execution of FFT on the plan (making the actual DFT of the kernels)

  K_out = out; // Saving this output into an array for kernel transforms

#ifdef PRINT   
  printf("\n Coefficients of the transformed kernel array: \n\n");
  for(i = 0; i < (NPOINTS/2+1); i++){
      printf("%d %11.7f %11.7f\n", i, creal(out[i]), cimag(out[i]));
    }
#endif

  // Inverse transform of the product of the two transforms:

  fftw_complex *inv_in;
  double *inv_out;
  fftw_plan inv_plan;

  // Allocate memory for storing an array which is the product of the two transforms
  inv_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*NPOINTS); 

  // Initializing the array for making the inverse transform (taking the array product)
  printf("\n Coefficients of the product array: \n\n");
  for(i = 0; i < (NPOINTS/2+1); i++){ 
    inv_in[i] = ld_out[i]*K_out[i]; 
#ifdef PRINT
    printf("%d %11.7f %11.7f\n", i, creal(inv_in[i]), cimag(inv_in[i]));
#endif 
  }

  inv_out = (double*) fftw_malloc(sizeof(double)*NPOINTS); // Initializing the output
  inv_plan = fftw_plan_dft_c2r_1d(NPOINTS, inv_in, inv_out, FFTW_ESTIMATE); // Inverse transform
  
  fftw_execute(inv_plan);
  

  FILE *of; // Declaring a file for saving the output/result
  of = fopen("data.dat","w"); // Initializing this file

  // Print the coefficients of the inverse DFT function (the output)
  printf("Inverse Output:\n\n"); 
  for(i = 0; i < NPOINTS; i++) {
    inv_out[i] = -G * STEP * 1/((double) NPOINTS) * inv_out[i];  // Normalizing the output
#ifdef PRINT
    printf("%d %11.7f \n", i, inv_out[i]); // Print coefficients in two seperate columns
#endif
    // Write the numerical potential, analytical potential, a counter, and the relative difference
    // between the two potentials, to a file (a column for each):
    fprintf(of,"%f %f %f %f \n", x[i], fabs(inv_out[i]), fabs(V_A[i]),
	    fabs((inv_out[i]-V_A[i])/V_A[i])); 
  }
  fclose(of); // Closing the file

  fftw_destroy_plan(plan); // Deallocating the forward plan and associated memory 
  fftw_free(in);
  fftw_free(out);
  fftw_free(ld_out); 

  fftw_destroy_plan(inv_plan); // Deallocating the backward plan and associated memory
  fftw_free(inv_in);
  fftw_free(inv_out);
  
 
  fftw_cleanup(); // Accumulated wisdom and a list of algorithms available in the current configuration become deallocated and FFTW is reset to its pristine state it was when the program started

  return 0;
}
