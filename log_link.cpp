#include <cmath>

#include <gsl/gsl_blas.h>

#include "log_link.h"

using namespace std;

void log_link::init_mov(gsl_vector * mov,gsl_vector * y)
{
	size_t n=y->size;
	for (int i = 0; i < n; ++i)
	{
		if (gsl_vector_get(y,i)==0)
		{
			gsl_vector_set(mov,i,0.01);
		}
		
		else
		{
			gsl_vector_set(mov,i,gsl_vector_get(y,i));
		}
	}
}

void log_link::compute_mov(gsl_matrix * Xv,gsl_vector *bv, gsl_vector * mov, gsl_vector * offset)
{
	size_t n= mov->size;
	size_t p= Xv->size2;
	gsl_matrix * fit = gsl_matrix_calloc(n,1);
	gsl_matrix *B= gsl_matrix_calloc(p,1);
	
	for (int i = 0; i < p; ++i)
	{
		gsl_matrix_set(B,i,0,gsl_vector_get(bv,i));
	}
    
    gsl_blas_dgemm(CblasNoTrans,CblasNoTrans,1,Xv,B,0,fit);
    for (int i = 0; i < n; ++i)
    {
    	gsl_vector_set(mov,i,exp(gsl_matrix_get(fit,i,0)+gsl_vector_get(offset,i)));
    }

    gsl_matrix_free(B);
    gsl_matrix_free(fit);

}



void log_link::compute_wieghts(gsl_vector * mov,gsl_vector * w)
{
	size_t n=mov->size;
	for (int i = 0; i < n; ++i)
	{
		gsl_vector_set(w,i,gsl_vector_get(mov,i));
	}
}

void log_link::compute_z(gsl_vector * mov, gsl_vector * z, gsl_vector * y,gsl_vector * offset)
{
	size_t n= mov->size ;
	double mv_i,val;
	for (int i = 0; i < n; ++i)
	{
		mv_i=gsl_vector_get(mov,i);
		gsl_vector_set(z,i,(log(mv_i)+(mv_i-gsl_vector_get(y,i))*(1/mv_i)+gsl_vector_get(offset,i)));
	}
}

double log_link::compute_dispersion(gsl_vector * y, gsl_matrix * Xv,
				   gsl_vector * bv, gsl_vector * offset,
				   gsl_vector * mov, double rank)
{
	compute_mov(Xv,bv,mov,offset);
	double wtss =0.0,mv_i,psi;
	size_t n=y->size;
	for(size_t i= 0; i < n;i++)
	{
		mv_i=gsl_vector_get(mov,i);
		wtss+= pow(gsl_vector_get(y,i)-mv_i,2)/mv_i;
	}
	psi=(1/(n-rank))*wtss;
	return psi;
}
int main(int argc, char const *argv[])
{
	/* code */
	return 0;
}