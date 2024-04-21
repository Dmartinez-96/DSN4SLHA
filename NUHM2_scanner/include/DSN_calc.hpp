// DSN_CALC_HPP

#ifndef DSN_CALC_HPP
#define DSN_CALC_HPP

#include <vector>
#include <string>

double DSN_calc(std::vector<double> GUT_boundary_conditions,
                double current_mZ2, double current_logQSUSY,
                double current_logQGUT);

#endif