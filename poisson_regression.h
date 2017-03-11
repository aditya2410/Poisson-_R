#ifndef _POISSON_H_
#define _POISSON_H_ 


#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <vector>
#include "log_link.h"


class poisson
{
    private:
        gsl_vector * y;      // response vector
        gsl_matrix * X;      // design matrix
        gsl_vector * offset; // offset vector
        bool free_data;      // depends on load_data() or set_data()
        size_t n;            // sample size;
        size_t p;            // number of parameters (including intercept)
        size_t rank;         // of X (useful for p-values)

        gsl_vector * bv;     // vector of estimated effect sizes
        gsl_matrix * VB;     // covariance matrix of estimated effect sizes
        double psi;          // dispersion
        log_link * link;
    public:
        
        poisson();
        ~poisson();

        void load_data(const std::vector<double> &yv,
         const std::vector<std::vector<double> > &Xv,
         const std::vector<double> &offv);
        void set_data(const std::vector<double> &yv,
         const std::vector<std::vector<double> > &Xv, 
         const std::vector<double> &offv);
        void fit_model();
        std::vector<double> get_coef();
        std::vector<double> get_stderr();
        size_t get_rank_X() { return rank; };
        double get_dispersion() { return psi; };
      
    private:
        void compute_variance(gsl_vector *w);
      
}; 
//Xv should not cantain the intercept
#endif



