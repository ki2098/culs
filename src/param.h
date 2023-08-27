#ifndef _PARAM_H_
#define _PARAM_H_ 1

const double kappa     = 1e3;
const double TL        = 0;
const double TR        = 1e2;
const double sor_omega = 1.2;

const double L         = 1;
const int    N         = 1000;

const int    restart = 10;

const unsigned int n_threads = 128;
const unsigned int n_blocks  = (N + n_threads - 1) / n_threads;

#endif