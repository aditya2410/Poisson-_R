#include "poisson_regression.h"

#include <cstring>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include "log_link.h"

using namespace  std;


poisson::poisson()
{
    bv = 0;
    VB = 0;
    link=new log_link();
}

poisson::~poisson()
{
    if(free_data)
    {
        gsl_vector_free(y);
        gsl_matrix_free(X);
        gsl_vector_free(offset);
    }

    if(bv !=0)
        gsl_vector_free(bv);

    if(VB != 0)
        gsl_matrix_free(VB);
}

void poisson::load_data(const std::vector<double> &yv,
    const std::vector<std::vector<double> > &Xv, 
    const std::vector<double> &offv)
{
    free_data = true;
    p=1+Xv.size();
    n=yv.size();
    y=gsl_vector_calloc(n);
    X=gsl_matrix_calloc(n,p);
    offset=gsl_vector_calloc(n);
    for (int i = 0; i < n; ++i)
    {
        gsl_matrix_set(X,i,0,1.0);
        gsl_vector_set(y,i,yv[i]);
        for (int j = 1; j < p; ++j)
        {
            gsl_matrix_set(X,i,j,Xv[j-1][i]);
        }

    }
    

    if (! offv.empty())
    {
        for (int i = 0; i < n; ++i)
        {
            gsl_vector_set(offset,i,offv[i]); 
        }
       
    }
    
}

void poisson::set_data(const std::vector<double> & yv,
    const std::vector<std::vector<double> > & Xv, 
    const std::vector<double> & offv)
{
    free_data=true;
    p=Xv.size();
    n=yv.size();
    y=gsl_vector_calloc(n);
    X=gsl_matrix_calloc(n,p);
    offset=gsl_vector_calloc(n);
    for (int i = 0; i < n; ++i)
    {
        gsl_vector_set(y,i,yv[i]);
        for (int j = 0; j < p; ++j)
        {
            gsl_matrix_set(X,i,j,Xv[j][i]);
        }

    }
    

    if (! offv.empty())
    {
        for (int i = 0; i < n; ++i)
        {
            gsl_vector_set(offset,i,offv[i]); 
        }
       
    }

}

void poisson::fit_model()
{
    gsl_vector *mov =gsl_vector_calloc(n);
    link->init_mov(mov,y);
    gsl_vector *z=gsl_vector_calloc(n);
    link->compute_z(mov,z,y,offset);
    gsl_vector *w=gsl_vector_calloc(n);
    link->compute_wieghts(mov,w);
    bv=gsl_vector_calloc(p);
    gsl_matrix *cov =gsl_matrix_alloc(p,p);
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc (n, p);
    double old_chisq = -1, chisq;
    gsl_multifit_wlinear_tsvd(X, w, z, GSL_DBL_EPSILON,bv, cov,&chisq,&rank, work);

    while(true)
    {
        if(fabs(chisq - old_chisq) < 1e-6)
        { // check convergence
            psi = link->compute_dispersion(y, X, bv, offset, mov, rank);
            compute_variance(w);
            break;
        }
        old_chisq=chisq;
        link->compute_mov(X,bv,mov,offset);
        link->compute_z(mov,z,y,offset);
        link->compute_wieghts(mov,w);
        
        gsl_multifit_wlinear_tsvd(X, w, z, GSL_DBL_EPSILON, bv, cov, &chisq,&rank, work);


    }
    gsl_vector_free(mov);
    gsl_vector_free(z);
    gsl_vector_free(w);
    gsl_matrix_free(cov);
    gsl_multifit_linear_free(work);
}

void poisson::compute_variance(gsl_vector * w)
{
  if(VB != 0)
    gsl_matrix_free(VB);
  
  VB = gsl_matrix_calloc(p, p);
  gsl_matrix * W = gsl_matrix_calloc(n, n);
  for(size_t i = 0; i < n; ++i)
    gsl_matrix_set(W, i, i, gsl_vector_get(w, i));
  
  gsl_matrix * t1 = gsl_matrix_calloc(p, n);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1, X, W, 0, t1);
  
  gsl_matrix * t2 = gsl_matrix_calloc(p, p);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, t1, X, 0, t2);
  
  // invert t2
  int ss;
  gsl_permutation * pp = gsl_permutation_alloc(p);
  gsl_linalg_LU_decomp(t2, pp, &ss);
  gsl_linalg_LU_invert(t2, pp, VB);
  
  gsl_matrix_scale(VB, psi); 
  
  gsl_matrix_free(W);
  gsl_matrix_free(t1);
  gsl_matrix_free(t2);
  gsl_permutation_free(pp);
}

vector<double> poisson::get_coef()
{
  vector<double> coev;
  for(size_t i=0; i < p; ++i)
    coev.push_back(gsl_vector_get(bv, i));
  return coev;
}

vector<double> poisson::get_stderr()
{
  vector<double> sev;
  for(size_t i = 0; i < p; ++i)
    sev.push_back(sqrt(gsl_matrix_get(VB, i, i)));
  return sev;
}


