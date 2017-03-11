#ifndef _LOGLINK_H
#define _LOGLINK_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

class log_link
{
public:
	void init_mov(gsl_vector * mov, gsl_vector * y);
	void compute_z(gsl_vector * y, gsl_vector * mov, 
		gsl_vector* z, gsl_vector * offset);
	void compute_wieghts(gsl_vector * w,gsl_vector *mov);
	void compute_mov(gsl_matrix * Xv,gsl_vector *bv,
	         gsl_vector * mov, gsl_vector * offsetv);
	double compute_dispersion(gsl_vector * y, gsl_matrix * Xv,
				   gsl_vector * bv, gsl_vector * offset,
				   gsl_vector * mov, double rank);
	~log_link(){};
};
#endif