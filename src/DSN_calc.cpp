#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>
#include <boost/math/special_functions/next.hpp>
#include "DSN_calc.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "MSSM_RGE_solver_with_U3Q3finder.hpp"
#include "mZ_numsolver.hpp"
#include "radcorr_calc.hpp"
#include "tree_mass_calc.hpp"
#include "EWSB_loop.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;

double signum(double x) {
    if (x < 0) {
        return -1.0;
    } else if (x > 0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

double soft_prob_calc(double x, int nPower) {
    return ((0.5 * x / (static_cast<double>(nPower) + 1.0))
            * ((signum(x) * (pow(x, nPower) - pow((-1.0) * x, nPower))) + pow(x, nPower) + pow((-1.0) * x, nPower)));
}

bool EWSB_Check(vector<double>& weak_boundary_conditions, vector<double>& radiat_correcs) {
    bool checkifEWSB = true;

    if (abs(2.0 * weak_boundary_conditions[42]) > abs((2.0 * pow(weak_boundary_conditions[6], 2.0)) + weak_boundary_conditions[25] + radiat_correcs[0] + weak_boundary_conditions[26] + radiat_correcs[1])) {
        // std::cout << "Scalar pot'l UFB at loop-level." << endl;
        checkifEWSB = false;
    }
    return checkifEWSB;
}

bool CCB_Check(vector<double>& weak_boundary_conditions) {
    bool checkifNoCCB = true;
    for (int i = 27; i < 42; ++i) {
        if (weak_boundary_conditions[i] < 0) {
            // std::cout << "CCB minima" << endl;
            checkifNoCCB = false;
        }
    }
    return checkifNoCCB;
}

vector<double> single_var_deriv_approxes(vector<double>& original_GUT_conditions, double& fixed_mZ2_val, int idx_to_shift, double& logQSUSYval, double& logQGUTval) {
    double p_orig, h_p, p_plus, p_minus, p_plusplus, p_minusminus;
    if (idx_to_shift == 42) {
        p_orig = original_GUT_conditions[idx_to_shift] / original_GUT_conditions[6];
        h_p = max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), 0.2), 1.0e-6);
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
    //    p_plusplus = p_plus + h_p;
    //    p_minusminus = p_minus - h_p;
    }
    else {
        p_orig = original_GUT_conditions[idx_to_shift];
        h_p = max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), 0.2), 1.0e-6);
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
    //    p_plusplus = p_plus + h_p;
    //    p_minusminus = p_minus - h_p;
    }

    vector<double> newmZ2GUTs_plus = original_GUT_conditions;
    //vector<double> newmZ2GUTs_plusplus = original_GUT_conditions;
    vector<double> newtanbGUTs_plus = original_GUT_conditions;
    //vector<double> newtanbGUTs_plusplus = original_GUT_conditions;
    vector<double> newmZ2GUTs_minus = original_GUT_conditions;
    //vector<double> newmZ2GUTs_minusminus = original_GUT_conditions;
    vector<double> newtanbGUTs_minus = original_GUT_conditions;
    //vector<double> newtanbGUTs_minusminus = original_GUT_conditions;

    double tanb_orig = original_GUT_conditions[43];
    double h_tanb = pow(10.25 * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), 0.2);
    //std::cout << "tanb step size = " << h_tanb << endl;
    newtanbGUTs_plus[43] = tanb_orig + h_tanb;
    newtanbGUTs_minus[43] = tanb_orig - h_tanb;
    //newtanbGUTs_plusplus[43] = tanb_orig + h_tanb + h_tanb;
    //newtanbGUTs_minusminus[43] = tanb_orig - h_tanb - h_tanb;

    vector<double> weaksolstanb_plus = solveODEs(newtanbGUTs_plus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolstanb_minus = solveODEs(newtanbGUTs_minus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    //vector<double> weaksolstanb_plusplus = solveODEs(newtanbGUTs_plusplus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    //vector<double> weaksolstanb_minusminus = solveODEs(newtanbGUTs_minusminus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));

    vector<double> TanbPlusRadCorrs = radcorr_calc(weaksolstanb_plus, exp(logQSUSYval), fixed_mZ2_val);
    vector<double> TanbMinusRadCorrs = radcorr_calc(weaksolstanb_minus, exp(logQSUSYval), fixed_mZ2_val);
    //vector<double> TanbPlusPlusRadCorrs = radcorr_calc(weaksolstanb_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    //vector<double> TanbMinusMinusRadCorrs = radcorr_calc(weaksolstanb_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    double tanb_mZ2_plus = (1.0 / 2.0) - (((weaksolstanb_plus[26] + TanbPlusRadCorrs[1] - ((weaksolstanb_plus[25] + TanbPlusRadCorrs[0]) * pow(weaksolstanb_plus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolstanb_plus[43], 2.0) - 1.0))) - (pow(weaksolstanb_plus[6], 2.0) / fixed_mZ2_val));
    double tanb_mZ2_minus = (1.0 / 2.0) - (((weaksolstanb_minus[26] + TanbMinusRadCorrs[1] - ((weaksolstanb_minus[25] + TanbMinusRadCorrs[0]) * pow(weaksolstanb_minus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolstanb_minus[43], 2.0) - 1.0))) - (pow(weaksolstanb_minus[6], 2.0) / fixed_mZ2_val));
    //double tanb_mZ2_plusplus = (1.0 / 2.0) - (((weaksolstanb_plusplus[26] + TanbPlusPlusRadCorrs[1] - ((weaksolstanb_plusplus[25] + TanbPlusPlusRadCorrs[0]) * pow(weaksolstanb_plusplus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolstanb_plusplus[43], 2.0) - 1.0))) - (pow(weaksolstanb_plusplus[6], 2.0) / fixed_mZ2_val));
    //double tanb_mZ2_minusminus = (1.0 / 2.0) - (((weaksolstanb_minusminus[26] + TanbMinusMinusRadCorrs[1] - ((weaksolstanb_minusminus[25] + TanbMinusMinusRadCorrs[0]) * pow(weaksolstanb_minusminus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolstanb_minusminus[43], 2.0) - 1.0))) - (pow(weaksolstanb_minusminus[6], 2.0) / fixed_mZ2_val));
    
    double tanb_tanb_plus = weaksolstanb_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolstanb_plus[42] / (weaksolstanb_plus[25] + TanbPlusRadCorrs[0] + weaksolstanb_plus[26] + TanbPlusRadCorrs[1] + (2.0 * pow(weaksolstanb_plus[6], 2.0))))));
    double tanb_tanb_minus = weaksolstanb_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolstanb_minus[42] / (weaksolstanb_minus[25] + TanbMinusRadCorrs[0] + weaksolstanb_minus[26] + TanbMinusRadCorrs[1] + (2.0 * pow(weaksolstanb_minus[6], 2.0))))));
    //double tanb_tanb_plusplus = weaksolstanb_plusplus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolstanb_plusplus[42] / (weaksolstanb_plusplus[25] + TanbPlusPlusRadCorrs[0] + weaksolstanb_plusplus[26] + TanbPlusPlusRadCorrs[1] + (2.0 * pow(weaksolstanb_plusplus[6], 2.0))))));
    //double tanb_tanb_minusminus = weaksolstanb_minusminus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolstanb_minusminus[42] / (weaksolstanb_minusminus[25] + TanbMinusMinusRadCorrs[0] + weaksolstanb_minusminus[26] + TanbMinusMinusRadCorrs[1] + (2.0 * pow(weaksolstanb_minusminus[6], 2.0))))));

    if (idx_to_shift == 6) {
        newmZ2GUTs_plus[42] = original_GUT_conditions[42] * p_plus / original_GUT_conditions[6];
    //    newmZ2GUTs_plusplus[42] = original_GUT_conditions[42] * p_plusplus / original_GUT_conditions[6];
        newmZ2GUTs_minus[42] = original_GUT_conditions[42] * p_minus / original_GUT_conditions[6];
    //    newmZ2GUTs_minusminus[42] = original_GUT_conditions[42] * p_minusminus / original_GUT_conditions[6];
        newmZ2GUTs_plus[6] = p_plus;
    //    newmZ2GUTs_plusplus[6] = p_plusplus;
        newmZ2GUTs_minus[6] = p_minus;
    //    newmZ2GUTs_minusminus[6] = p_minusminus;
    } else if (idx_to_shift == 42) {
        newmZ2GUTs_plus[42] = original_GUT_conditions[6] * p_plus;
    //    newmZ2GUTs_plusplus[42] = original_GUT_conditions[6] * p_plusplus;
        newmZ2GUTs_minus[42] = original_GUT_conditions[6] * p_minus;
    //    newmZ2GUTs_minusminus[42] = original_GUT_conditions[6] * p_minusminus;
    } else {
        newmZ2GUTs_plus[idx_to_shift] = p_plus;
        newmZ2GUTs_minus[idx_to_shift] = p_minus;
    //    newmZ2GUTs_plusplus[idx_to_shift] = p_plusplus;
    //    newmZ2GUTs_minusminus[idx_to_shift] = p_minusminus;
    }

    vector<double> weaksolsmZ2_plus = solveODEs(newmZ2GUTs_plus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    //vector<double> weaksolsmZ2_plusplus = solveODEs(newmZ2GUTs_plusplus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    vector<double> weaksolsmZ2_minus = solveODEs(newmZ2GUTs_minus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    //vector<double> weaksolsmZ2_minusminus = solveODEs(newmZ2GUTs_minusminus, logQGUTval, logQSUSYval, copysign(1.0e-6, logQSUSYval - logQGUTval));
    
    vector<double> pPlusRadCorrs = radcorr_calc(weaksolsmZ2_plus, exp(logQSUSYval), fixed_mZ2_val);
    //vector<double> pPlusPlusRadCorrs = radcorr_calc(weaksolsmZ2_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    vector<double> pMinusRadCorrs = radcorr_calc(weaksolsmZ2_minus, exp(logQSUSYval), fixed_mZ2_val);
    //vector<double> pMinusMinusRadCorrs = radcorr_calc(weaksolsmZ2_minusminus, exp(logQSUSYval), fixed_mZ2_val);

    double p_mZ2_plus = (1.0 / 2.0) - (((weaksolsmZ2_plus[26] + pPlusRadCorrs[1] - ((weaksolsmZ2_plus[25] + pPlusRadCorrs[0]) * pow(weaksolsmZ2_plus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolsmZ2_plus[43], 2.0) - 1.0))) - (pow(weaksolsmZ2_plus[6], 2.0) / fixed_mZ2_val));
    //double p_mZ2_plusplus = (1.0 / 2.0) - (((weaksolsmZ2_plusplus[26] + pPlusPlusRadCorrs[1] - ((weaksolsmZ2_plusplus[25] + pPlusPlusRadCorrs[0]) * pow(weaksolsmZ2_plusplus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolsmZ2_plusplus[43], 2.0) - 1.0))) - (pow(weaksolsmZ2_plusplus[6], 2.0) / fixed_mZ2_val));
    double p_mZ2_minus = (1.0 / 2.0) - (((weaksolsmZ2_minus[26] + pMinusRadCorrs[1] - ((weaksolsmZ2_minus[25] + pMinusRadCorrs[0]) * pow(weaksolsmZ2_minus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolsmZ2_minus[43], 2.0) - 1.0))) - (pow(weaksolsmZ2_minus[6], 2.0) / fixed_mZ2_val));
    //double p_mZ2_minusminus = (1.0 / 2.0) - (((weaksolsmZ2_minusminus[26] + pMinusMinusRadCorrs[1] - ((weaksolsmZ2_minusminus[25] + pMinusMinusRadCorrs[0]) * pow(weaksolsmZ2_minusminus[43], 2.0))) / (fixed_mZ2_val * (pow(weaksolsmZ2_minusminus[43], 2.0) - 1.0))) - (pow(weaksolsmZ2_minusminus[6], 2.0) / fixed_mZ2_val));
    double p_tanb_plus = weaksolsmZ2_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolsmZ2_plus[42] / (weaksolsmZ2_plus[25] + pPlusRadCorrs[0] + weaksolsmZ2_plus[26] + pPlusRadCorrs[1] + (2.0 * pow(weaksolsmZ2_plus[6], 2.0))))));
    //double p_tanb_plusplus = weaksolsmZ2_plusplus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolsmZ2_plusplus[42] / (weaksolsmZ2_plusplus[25] + pPlusPlusRadCorrs[0] + weaksolsmZ2_plusplus[26] + pPlusPlusRadCorrs[1] + (2.0 * pow(weaksolsmZ2_plusplus[6], 2.0))))));
    double p_tanb_minus = weaksolsmZ2_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolsmZ2_minus[42] / (weaksolsmZ2_minus[25] + pMinusRadCorrs[0] + weaksolsmZ2_minus[26] + pMinusRadCorrs[1] + (2.0 * pow(weaksolsmZ2_minus[6], 2.0))))));
    //double p_tanb_minusminus = weaksolsmZ2_minusminus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolsmZ2_minusminus[42] / (weaksolsmZ2_minusminus[25] + pMinusMinusRadCorrs[0] + weaksolsmZ2_minusminus[26] + pMinusMinusRadCorrs[1] + (2.0 * pow(weaksolsmZ2_minusminus[6], 2.0))))));

    /* Order of derivatives:
        (0: d(mZ^2 eqn)/dp,
         1: d(mZ^2 eqn)/d(tanb),
         2: d(tanb eqn)/dp, 
         3: d(tanb eqn)/d(tanb))
    */
    vector<double> evaluated_derivs = {((0.5 * p_mZ2_plus) - (0.5 * p_mZ2_minus)) / h_p,//(((-1.0) * p_mZ2_plusplus / 12.0) + (2.0 * p_mZ2_plus / 3.0) - (2.0 * p_mZ2_minus / 3.0) + (p_mZ2_minusminus / 12.0)) / (h_p),
                                       ((0.5 * tanb_mZ2_plus) - (0.5 * tanb_mZ2_minus)) / h_tanb,//(((-1.0) * tanb_mZ2_plusplus / 12.0) + (2.0 * tanb_mZ2_plus / 3.0) - (2.0 * tanb_mZ2_minus / 3.0) + (tanb_mZ2_minusminus / 12.0)) / (h_tanb),
                                       ((0.5 * p_tanb_plus) - (0.5 * p_tanb_minus)) / h_p,//(((-1.0) * p_tanb_plusplus / 12.0) + (2.0 * p_tanb_plus / 3.0) - (2.0 * p_tanb_minus / 3.0) + (p_tanb_minusminus / 12.0)) / (h_p),
                                       ((0.5 * tanb_tanb_plus) - (0.5 * tanb_tanb_minus)) / h_tanb};//(((-1.0) * tanb_tanb_plusplus / 12.0) + (2.0 * tanb_tanb_plus / 3.0) - (2.0 * tanb_tanb_minus / 3.0) + (tanb_tanb_minusminus / 12.0)) / (h_tanb)};
    return evaluated_derivs;
}

vector<double> get_F_G_vals(vector<double>& GUT_BCs, double& curr_mZ2, double& curr_logQSUSY, double& curr_logQGUT, int derivIndex) {
    //vector<double> derivatives = single_var_deriv_approxes(GUT_BCs, curr_mZ2, derivIndex, curr_logQSUSY, curr_logQGUT);
            
    vector<double> weaksolution = solveODEs(GUT_BCs, curr_logQGUT, curr_logQSUSY, copysign(1.0e-6, (curr_logQSUSY - curr_logQGUT)));
    vector<double> RadiatCorrs = radcorr_calc(weaksolution, exp(curr_logQSUSY), curr_mZ2);
    
    double myF = (1.0 / 2.0) - (((weaksolution[26] + RadiatCorrs[1] - ((weaksolution[25] + RadiatCorrs[0]) * pow(weaksolution[43], 2.0))) / (curr_mZ2 * (pow(weaksolution[43], 2.0) - 1.0))) - (weaksolution[6] * weaksolution[6] / curr_mZ2));
    double myG = weaksolution[43] - tan(0.5 * (M_PI - asin(2.0 * weaksolution[42] / (weaksolution[25] + RadiatCorrs[0] + weaksolution[26] + RadiatCorrs[1] + (2.0 * pow(weaksolution[6], 2.0))))));
    return {myF, myG};
}

vector<double> DSN_B_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT) {
    double t_target = log(500.0);
    vector<double> BnewGUTs_plus = GUT_boundary_conditions;
    vector<double> BnewGUTs_minus = GUT_boundary_conditions;
    double BcurrentlogQGUT = current_logQGUT;
    double BcurrentlogQSUSY = current_logQSUSY;
    double BnewlogQGUT = current_logQGUT;
    double BnewlogQSUSY = current_logQSUSY;
    double Bnew_mZ2plus = current_mZ2;
    double Bnew_mZ2minus = current_mZ2;

    double Bplus = BnewGUTs_plus[42] / BnewGUTs_plus[6];
    double newBplus = Bplus;

    double Bminus = BnewGUTs_minus[42] / BnewGUTs_minus[6];
    double newBminus = Bminus;

    bool BminusNoCCB = true;
    bool BminusEWSB = true;
    bool BplusNoCCB = true;
    bool BplusEWSB = true;
    double Bcurr_tanbGUT = GUT_boundary_conditions[43];
    double Bnew_tanbGUT = Bcurr_tanbGUT;
    double Bcurr_mZ2plus = current_mZ2;
    double Bcurr_mZ2minus = current_mZ2;
    double muGUT_original = GUT_boundary_conditions[6];
    // First, compute width of ABDS window
    double lambdaB = 0.5;
    double B_least_Sq_Tol = 1.0e-4;
    double prev_fB = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    
    while ((BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > (45.5938 * 45.5938)) && (Bnew_mZ2minus < (364.7504 * 364.7504)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2minus *= 0.99;
        // std::cout << "New mZ = " << sqrt(Bnew_mZ2minus) << endl;
        // std::cout << "New B = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
        BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BminusEWSB == true) {
            BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BminusEWSB == true) {
        //     BminusEWSB = BFB_check(checkweaksols);
        // }
        BminusNoCCB = CCB_Check(checkweaksols);
        if (!(BminusEWSB) || !(BminusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BminusNoCCB = false;
            }
        }
        if (!(BminusNoCCB)) {
            break;
        }                
        vector<double> BoldGUTs_minus = BnewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_minus, Bnew_mZ2minus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(BnewlogQSUSY), Bnew_mZ2minus);
            
            double FB = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (Bnew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / Bnew_mZ2minus));
            double GB = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_minus[43] = BnewGUTs_minus[43] - DeltaTanb;
            BnewGUTs_minus[42] = ((BnewGUTs_minus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_minus[42]) || (isnan(BnewGUTs_minus[43])))) {
                BminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_minus[6] = BoldGUTs_minus[6];
            BnewGUTs_minus[42] = BoldGUTs_minus[42];
            BnewGUTs_minus[43] = BoldGUTs_minus[43];
            break;
        }
        if ((BnewGUTs_minus[43] < 3.0) || (BnewGUTs_minus[43] > 60.0)) {
            BminusEWSB = false;
        }
    }
    // std::cout << "B(ABDS, minus) = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
    double B_GUT_minus = BnewGUTs_minus[42] / BnewGUTs_minus[6];
    if (B_GUT_minus == GUT_boundary_conditions[42] / GUT_boundary_conditions[6]) {
        B_GUT_minus = copysign(boost::math::float_prior(abs(B_GUT_minus)), B_GUT_minus);
    }
    Bcurr_tanbGUT = GUT_boundary_conditions[43];
    Bnew_tanbGUT = Bcurr_tanbGUT;
    BcurrentlogQGUT = current_logQGUT;
    BcurrentlogQSUSY = current_logQSUSY;

    while ((BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > (45.5938 * 45.5938)) && (Bnew_mZ2plus < (364.7504 * 364.7504)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2plus *= 1.01;
        //std::cout << "New mZ = " << sqrt(Bnew_mZ2plus) << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
        BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BplusEWSB == true) {
            BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BplusEWSB == true) {
        //     BplusEWSB = BFB_check(checkweaksols);
        // }
        BplusNoCCB = CCB_Check(checkweaksols);
        if (!(BplusEWSB) || !(BplusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BplusNoCCB = false;
            }
        }
        if (!(BplusNoCCB)) {
            break;
        }             
        vector<double> BoldGUTs_plus = BnewGUTs_plus;   
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_plus, Bnew_mZ2plus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_plus = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsPlus = radcorr_calc(weaksol_plus, exp(BnewlogQSUSY), Bnew_mZ2plus);
            
            double FB = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsPlus[1] - ((weaksol_plus[25] + RadCorrsPlus[0]) * pow(weaksol_plus[43], 2.0))) / (Bnew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / Bnew_mZ2plus));
            double GB = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsPlus[0] + weaksol_plus[26] + RadCorrsPlus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_plus[43] = BnewGUTs_plus[43] - DeltaTanb;
            BnewGUTs_plus[42] = ((BnewGUTs_plus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_plus[42]) || (isnan(BnewGUTs_plus[43])))) {
                BplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_plus[6] = BoldGUTs_plus[6];
            BnewGUTs_plus[42] = BoldGUTs_plus[42];
            BnewGUTs_plus[43] = BoldGUTs_plus[43];
            break;
        }
        if ((BnewGUTs_plus[43] < 3.0) || (BnewGUTs_plus[43] > 60.0)) {
            BplusEWSB = false;
        }
    }
    // std::cout << "B(ABDS, plus) = " << BnewGUTs_plus[42] / BnewGUTs_plus[6] << endl;
    double B_GUT_plus = BnewGUTs_plus[42] / BnewGUTs_plus[6];
    if (B_GUT_plus == GUT_boundary_conditions[42] / GUT_boundary_conditions[6]) {
        B_GUT_plus = copysign(boost::math::float_next(abs(B_GUT_plus)), B_GUT_plus);
    }
    Bcurr_tanbGUT = BnewGUTs_minus[43];
    Bnew_tanbGUT = Bcurr_tanbGUT;
    BcurrentlogQGUT = current_logQGUT;
    BcurrentlogQSUSY = current_logQSUSY;

    std::cout << "ABDS window established for B variation." << endl;
    
    bool ABDSminuscheck = (BminusEWSB && BminusNoCCB); 
    bool ABDSpluscheck = (BplusEWSB && BplusNoCCB);
    double B_TOTAL_GUT_minus, B_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (B_GUT_minus <= B_GUT_plus) {
            B_TOTAL_GUT_minus = boost::math::float_prior(B_GUT_minus);
            B_TOTAL_GUT_plus = boost::math::float_next(B_GUT_plus);
        } else {
            B_TOTAL_GUT_minus = boost::math::float_next(B_GUT_minus);
            B_TOTAL_GUT_plus = boost::math::float_prior(B_GUT_plus);
        }

        std::cout << "General window established for B variation." << endl;

        return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
    }

    while ((BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > (1.0)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2minus *= 0.99;
        // std::cout << "New mZ = " << sqrt(Bnew_mZ2minus) << endl;
        // std::cout << "New B = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
        BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BminusEWSB == true) {
            BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BminusEWSB == true) {
        //     BminusEWSB = BFB_check(checkweaksols);
        // }
        BminusNoCCB = CCB_Check(checkweaksols);
        if (!(BminusEWSB) || !(BminusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BminusNoCCB = false;
            }
        }
        if (!(BminusNoCCB)) {
            break;
        }                
        vector<double> BoldGUTs_minus = BnewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_minus, Bnew_mZ2minus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(BnewGUTs_minus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(BnewlogQSUSY), Bnew_mZ2minus);
            
            double FB = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (Bnew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / Bnew_mZ2minus));
            double GB = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_minus[43] = BnewGUTs_minus[43] - DeltaTanb;
            BnewGUTs_minus[42] = ((BnewGUTs_minus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_minus[42]) || (isnan(BnewGUTs_minus[43])))) {
                BminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_minus[6] = BoldGUTs_minus[6];
            BnewGUTs_minus[42] = BoldGUTs_minus[42];
            BnewGUTs_minus[43] = BoldGUTs_minus[43];
            break;
        }
        if ((BnewGUTs_minus[43] < 3.0) || (BnewGUTs_minus[43] > 60.0)) {
            BminusEWSB = false;
        }
    }
    // std::cout << "B(total, minus) = " << BnewGUTs_minus[42] / BnewGUTs_minus[6] << endl;
    B_TOTAL_GUT_minus = BnewGUTs_minus[42] / BnewGUTs_minus[6];

    while ((BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > (1.0)))) {
        lambdaB = 0.5;
        B_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        Bnew_mZ2plus *= 1.01;
        //std::cout << "New mZ = " << sqrt(Bnew_mZ2plus) << endl;
        int numStepsDone = 0;
        vector<double> checkweaksols = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
        BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        if (BplusEWSB == true) {
            BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
        }
        // if (BplusEWSB == true) {
        //     BplusEWSB = BFB_check(checkweaksols);
        // }
        BplusNoCCB = CCB_Check(checkweaksols);
        if (!(BplusEWSB) || !(BplusNoCCB)) {
            break;
        }
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(BcurrentlogQSUSY), Bnew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                BplusNoCCB = false;
            }
        }
        if (!(BplusNoCCB)) {
            break;
        }             
        vector<double> BoldGUTs_plus = BnewGUTs_plus;   
        while ((numStepsDone < 100) && (curr_lsq_eval > B_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(BnewGUTs_plus, Bnew_mZ2plus, 42, BnewlogQSUSY, BnewlogQGUT);
            
            vector<double> weaksol_plus = solveODEs(BnewGUTs_plus, BcurrentlogQGUT, BcurrentlogQSUSY, copysign(1.0e-6, (BcurrentlogQSUSY - BcurrentlogQGUT)));
            vector<double> RadCorrsPlus = radcorr_calc(weaksol_plus, exp(BnewlogQSUSY), Bnew_mZ2plus);
            
            double FB = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsPlus[1] - ((weaksol_plus[25] + RadCorrsPlus[0]) * pow(weaksol_plus[43], 2.0))) / (Bnew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / Bnew_mZ2plus));
            double GB = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsPlus[0] + weaksol_plus[26] + RadCorrsPlus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GB) - (current_derivatives[2] * FB));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaBNum = ((current_derivatives[3] * FB) - (current_derivatives[1] * GB));
            double DeltaB = DeltaBNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaB, 2.0) + pow(DeltaTanb, 2.0);
            BnewGUTs_plus[43] = BnewGUTs_plus[43] - DeltaTanb;
            BnewGUTs_plus[42] = ((BnewGUTs_plus[42] / muGUT_original) - DeltaB) * muGUT_original;
            if ((isnan(BnewGUTs_plus[42]) || (isnan(BnewGUTs_plus[43])))) {
                BplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!BplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            BnewGUTs_plus[6] = BoldGUTs_plus[6];
            BnewGUTs_plus[42] = BoldGUTs_plus[42];
            BnewGUTs_plus[43] = BoldGUTs_plus[43];
            break;
        }
        if ((BnewGUTs_plus[43] < 3.0) || (BnewGUTs_plus[43] > 60.0)) {
            BplusEWSB = false;
        }
    }
    // std::cout << "B(total, plus) = " << BnewGUTs_plus[42] / BnewGUTs_plus[6] << endl;
    B_TOTAL_GUT_plus = BnewGUTs_plus[42] / BnewGUTs_plus[6];

    if ((abs(B_TOTAL_GUT_minus - B_GUT_minus) < 1.0e-12) && (abs(B_TOTAL_GUT_plus - B_GUT_plus) < 1.0e-12)) {
        if (B_GUT_minus <= B_GUT_plus) {
            B_TOTAL_GUT_minus = boost::math::float_prior(B_GUT_minus);
            B_TOTAL_GUT_plus = boost::math::float_next(B_GUT_plus);
        } else {
            B_TOTAL_GUT_minus = boost::math::float_next(B_GUT_minus);
            B_TOTAL_GUT_plus = boost::math::float_prior(B_GUT_plus);
        }
        std::cout << "General window established for B variation." << endl;

        return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
    }
    std::cout << "General window established for B variation." << endl;

    return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
}

vector<double> DSN_specific_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT, int SpecificIndex) {
    double t_target = log(500.0);
    vector<double> pinewGUTs_plus = GUT_boundary_conditions;
    vector<double> pinewGUTs_minus = GUT_boundary_conditions;
    double picurrentlogQGUT = current_logQGUT;
    double picurrentlogQSUSY = current_logQSUSY;
    double pinewlogQGUT = current_logQGUT;
    double pinewlogQSUSY = current_logQSUSY;
    double pinew_mZ2plus = current_mZ2;
    double pinew_mZ2minus = current_mZ2;
    bool piminusNoCCB = true;
    bool piminusEWSB = true;
    bool piplusNoCCB = true;
    bool piplusEWSB = true;
    string paramName;
    if (SpecificIndex == 3) {
        paramName = "M1";
    } else if (SpecificIndex == 4) {
        paramName = "M2";
    } else if (SpecificIndex == 5) {
        paramName = "M3";
    } else if (SpecificIndex == 16) {
        paramName = "a_t";
    } else if (SpecificIndex == 17) {
        paramName = "a_c";
    } else if (SpecificIndex == 18) {
        paramName = "a_u";
    } else if (SpecificIndex == 19) {
        paramName = "a_b";
    } else if (SpecificIndex == 20) {
        paramName = "a_s";
    } else if (SpecificIndex == 21) {
        paramName = "a_d";
    } else if (SpecificIndex == 22) {
        paramName = "a_tau";
    } else if (SpecificIndex == 23) {
        paramName = "a_mu";
    } else if (SpecificIndex == 24) {
        paramName = "a_e";
    } else if (SpecificIndex == 25) {
        paramName = "mHu^2";
    } else if (SpecificIndex == 26) {
        paramName = "mHd^2";
    } else if (SpecificIndex == 27) {
        paramName = "mQ1^2";
    } else if (SpecificIndex == 28) {
        paramName = "mQ2^2";
    } else if (SpecificIndex == 29) {
        paramName = "mQ3^2";
    } else if (SpecificIndex == 30) {
        paramName = "mL1^2";
    } else if (SpecificIndex == 31) {
        paramName = "mL2^2";
    } else if (SpecificIndex == 32) {
        paramName = "mL3^2";
    } else if (SpecificIndex == 33) {
        paramName = "mU1^2";
    } else if (SpecificIndex == 34) {
        paramName = "mU2^2";
    } else if (SpecificIndex == 35) {
        paramName = "mU3^2";
    } else if (SpecificIndex == 36) {
        paramName = "mD1^2";
    } else if (SpecificIndex == 37) {
        paramName = "mD2^2";
    } else if (SpecificIndex == 38) {
        paramName = "mD3^2";
    } else if (SpecificIndex == 39) {
        paramName = "mE1^2";
    } else if (SpecificIndex == 40) {
        paramName = "mE2^2";
    } else {
        paramName = "mE3^2";
    }

    double piplus = pinewGUTs_plus[SpecificIndex];
    double newpiplus = piplus;
    double tanbplus = pinewGUTs_plus[43];
    double newtanbplus = tanbplus;

    double piminus = pinewGUTs_minus[SpecificIndex];
    double newpiminus = piminus;
    double tanbminus = pinewGUTs_minus[43];
    double newtanbminus = tanbminus;

    // First compute width of ABDS window
    double lambdapi = 0.5;
    double pi_least_Sq_Tol = 1.0e-4;
    double prev_fpi = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    while ((piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > (45.5938 * 45.5938)) && (pinew_mZ2minus < (364.7504 * 364.7504)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2minus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_minus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2minus);
        piminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piminusEWSB == true) {
            piminusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piminusEWSB == true) {
        //     piminusEWSB = BFB_check(checkweaksols);
        // }
        piminusNoCCB = CCB_Check(checkweaksols);
        if (!(piminusEWSB) || !(piminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piminusNoCCB = false;
            }
        }
        if (!(piminusNoCCB)) {
            break;
        } 
        vector<double> oldSolutions = pinewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(pinewlogQSUSY), pinew_mZ2minus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (pinew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / pinew_mZ2minus));
            double GMi = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;
            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_minus[43] = pinewGUTs_minus[43] - DeltaTanb;
            pinewGUTs_minus[SpecificIndex] = pinewGUTs_minus[SpecificIndex] - DeltaPi;
            newpiminus = pinewGUTs_minus[SpecificIndex];
            if ((isnan(pinewGUTs_minus[SpecificIndex])) || (isnan(pinewGUTs_minus[43]))) {
                piminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_minus, pinew_mZ2minus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            piminusEWSB = false;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
        }
        if ((pinewGUTs_minus[43] < 3.0) || (pinewGUTs_minus[43] > 60.0)) {
            piminusEWSB = false;
        }
    }
    // std::cout << paramName << "(ABDS, minus) = " << pinewGUTs_minus[SpecificIndex] << endl; 
    double pi_GUT_minus = pinewGUTs_minus[SpecificIndex];
    if (pi_GUT_minus == GUT_boundary_conditions[SpecificIndex]) {
        pi_GUT_minus = copysign(boost::math::float_prior(abs(pi_GUT_minus)), pi_GUT_minus);
    }
    lambdapi = 0.5;
    pi_least_Sq_Tol = 1.0e-4;
    prev_fpi = std::numeric_limits<double>::max();
    while ((piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > (45.5938 * 45.5938)) && (pinew_mZ2plus < (364.7504 * 364.7504)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2plus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_plus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
        piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piplusEWSB == true) {
            piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piplusEWSB == true) {
        //     piplusEWSB = BFB_check(checkweaksols);
        // }
        piplusNoCCB = CCB_Check(checkweaksols);
        if (!(piplusEWSB) || !(piplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piplusNoCCB = false;
            }
        }
        if (!(piplusNoCCB)) {
            break;
        } 
                // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        vector<double> oldSolutions = pinewGUTs_plus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            //std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);

            vector<double> weaksol_plus = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(pinewlogQSUSY), pinew_mZ2plus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (pinew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / pinew_mZ2plus));
            double GMi = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_plus[43] = pinewGUTs_plus[43] - DeltaTanb;
            pinewGUTs_plus[SpecificIndex] = pinewGUTs_plus[SpecificIndex] - DeltaPi;
                        
            newpiplus = pinewGUTs_plus[SpecificIndex];
            if ((isnan(pinewGUTs_plus[SpecificIndex])) || (isnan(pinewGUTs_plus[43]))) {
                piplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
            break;
        }  
        if ((pinewGUTs_plus[43] < 3.0) || (pinewGUTs_plus[43] > 60.0)) {
            piplusEWSB = false;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_plus, pinew_mZ2plus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2) || (isnan(check_valid_solutions[0])) || (isnan(check_valid_solutions[1]))) {
            // std::cout << "Failed to converge" << endl;
            piplusEWSB = false;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
        }         
    }
    // std::cout << paramName << "(ABDS, plus) = " << pinewGUTs_plus[SpecificIndex] << endl; 
    double pi_GUT_plus = pinewGUTs_plus[SpecificIndex];
    if (pi_GUT_plus == GUT_boundary_conditions[SpecificIndex]) {
        pi_GUT_plus = copysign(boost::math::float_prior(abs(pi_GUT_plus)), pi_GUT_plus);
    }

    std::cout << "ABDS window established for " << paramName << " variation." << endl;

    bool ABDSminuscheck = (piminusEWSB && piminusNoCCB); 
    bool ABDSpluscheck = (piplusEWSB && piplusNoCCB);
    double pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (pi_GUT_minus <= pi_GUT_plus) {
            pi_TOTAL_GUT_minus = boost::math::float_prior(pi_GUT_minus);
            pi_TOTAL_GUT_plus = boost::math::float_next(pi_GUT_plus);
        } else {
            pi_TOTAL_GUT_minus = boost::math::float_next(pi_GUT_minus);
            pi_TOTAL_GUT_plus = boost::math::float_prior(pi_GUT_plus);
        }

        std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
    }

    while ((piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > (1.0)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2minus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_minus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2minus);
        piminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piminusEWSB == true) {
            piminusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        piminusNoCCB = CCB_Check(checkweaksols);
        if (!(piminusEWSB) || !(piminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piminusNoCCB = false;
            }
        }
        if (!(piminusNoCCB)) {
            break;
        } 
        vector<double> oldSolutions = pinewGUTs_minus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(pinewGUTs_minus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(pinewlogQSUSY), pinew_mZ2minus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (pinew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / pinew_mZ2minus));
            double GMi = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_minus[43] = pinewGUTs_minus[43] - DeltaTanb;
            pinewGUTs_minus[SpecificIndex] = pinewGUTs_minus[SpecificIndex] - DeltaPi;
            newpiminus = pinewGUTs_minus[SpecificIndex];
            if ((isnan(pinewGUTs_minus[SpecificIndex])) || (isnan(pinewGUTs_minus[43]))) {
                piminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_minus, pinew_mZ2minus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            piminusEWSB = false;
            pinewGUTs_minus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_minus[43] = oldSolutions[43];
        }
        if ((pinewGUTs_minus[43] < 3.0) || (pinewGUTs_minus[43] > 60.0)) {
            piminusEWSB = false;
        }
    }
    // std::cout << paramName << "(total, minus) = " << pinewGUTs_minus[SpecificIndex] << endl; 
    pi_TOTAL_GUT_minus = pinewGUTs_minus[SpecificIndex];

    while ((piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > (1.0)))) {
        lambdapi = 0.5;
        pi_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fpi = std::numeric_limits<double>::max();
        pinew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(pinew_mZ2plus) << endl;
        // std::cout << "New " << paramName << "(GUT) = " << pinewGUTs_plus[SpecificIndex] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
        piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (piplusEWSB == true) {
            piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
        }
        // if (piplusEWSB == true) {
        //     piplusEWSB = BFB_check(checkweaksols);
        // }
        piplusNoCCB = CCB_Check(checkweaksols);
        if (!(piplusEWSB) || !(piplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(picurrentlogQSUSY), pinew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                piplusNoCCB = false;
            }
        }
        if (!(piplusNoCCB)) {
            break;
        } 
                // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        vector<double> oldSolutions = pinewGUTs_plus;
        while ((numStepsDone < 100) && (curr_lsq_eval > pi_least_Sq_Tol)) {
            //std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(pinewGUTs_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY, pinewlogQGUT);

            vector<double> weaksol_plus = solveODEs(pinewGUTs_plus, picurrentlogQGUT, picurrentlogQSUSY, copysign(1.0e-6, (picurrentlogQSUSY - picurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(pinewlogQSUSY), pinew_mZ2plus);
            
            double FMi = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (pinew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / pinew_mZ2plus));
            double GMi = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMi) - (current_derivatives[2] * FMi));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaPiNum = ((current_derivatives[3] * FMi) - (current_derivatives[1] * GMi));
            double DeltaPi = DeltaPiNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaPi, 2.0) + pow(DeltaTanb, 2.0);
            pinewGUTs_plus[43] = pinewGUTs_plus[43] - DeltaTanb;
            pinewGUTs_plus[SpecificIndex] = pinewGUTs_plus[SpecificIndex] - DeltaPi;
                        
            newpiplus = pinewGUTs_plus[SpecificIndex];
            if ((isnan(pinewGUTs_plus[SpecificIndex])) || (isnan(pinewGUTs_plus[43]))) {
                piplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!piplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
            break;
        }  
        if ((pinewGUTs_plus[43] < 3.0) || (pinewGUTs_plus[43] > 60.0)) {
            piplusEWSB = false;
        }
        vector<double> check_valid_solutions = get_F_G_vals(pinewGUTs_plus, pinew_mZ2plus, current_logQSUSY, current_logQGUT, SpecificIndex);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2) || (isnan(check_valid_solutions[0])) || (isnan(check_valid_solutions[1]))) {
            // std::cout << "Failed to converge" << endl;
            piplusEWSB = false;
            pinewGUTs_plus[SpecificIndex] = oldSolutions[SpecificIndex];
            pinewGUTs_plus[43] = oldSolutions[43];
        }         
    }
    // std::cout << paramName << "(total, plus) = " << pinewGUTs_plus[SpecificIndex] << endl; 
    pi_TOTAL_GUT_plus = pinewGUTs_plus[SpecificIndex];

    if ((abs(pi_TOTAL_GUT_minus - pi_GUT_minus) < 1.0e-12) && (abs(pi_TOTAL_GUT_plus - pi_GUT_plus) < 1.0e-12)) {
        if (pi_GUT_minus <= pi_GUT_plus) {
            pi_TOTAL_GUT_minus = boost::math::float_prior(pi_GUT_minus);
            pi_TOTAL_GUT_plus = boost::math::float_next(pi_GUT_plus);
        } else {
            pi_TOTAL_GUT_minus = boost::math::float_next(pi_GUT_minus);
            pi_TOTAL_GUT_plus = boost::math::float_prior(pi_GUT_plus);
        }

        std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
    }

    std::cout << "General window established for " << paramName << " variation." << endl;

    return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
}

vector<double> DSN_mu_windows(vector<double>& GUT_boundary_conditions, double& current_mZ2, double& current_logQSUSY, double& current_logQGUT) {
    double t_target = log(500.0);
    vector<double> munewGUTs_plus = GUT_boundary_conditions;
    vector<double> munewGUTs_minus = GUT_boundary_conditions;
    double mucurrentlogQGUT = current_logQGUT;
    double mucurrentlogQSUSY = current_logQSUSY;
    double munewlogQGUT = current_logQGUT;
    double munewlogQSUSY = current_logQSUSY;
    double munew_mZ2plus = current_mZ2;
    double munew_mZ2minus = current_mZ2;
    bool muminusNoCCB = true;
    bool muminusEWSB = true;
    bool muplusNoCCB = true;
    bool muplusEWSB = true;

    double muplus = munewGUTs_plus[6];
    double newmuplus = muplus;
    double tanbplus = munewGUTs_plus[43];
    double newtanbplus = tanbplus;

    double muminus = munewGUTs_minus[6];
    double newmuminus = muminus;
    double tanbminus = munewGUTs_minus[43];
    double newtanbminus = tanbminus;
    double BGUT_original = GUT_boundary_conditions[42] / GUT_boundary_conditions[6];

    // First compute width of ABDS window
    double lambdaMu = 0.5;
    double Mu_least_Sq_Tol = 1.0e-4;
    double prev_fmu = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    // Mu convergence becomes bad when mu is small (i.e. < 10 GeV), so cutoff at abs(mu) = 10 GeV
    while ((muminusEWSB) && (muminusNoCCB) && (abs(munewGUTs_minus[6]) > 10.0) && ((munew_mZ2minus > (45.5938 * 45.5938)) && (munew_mZ2minus < (364.7504 * 364.7504)))) {
        lambdaMu = 0.5;
        Mu_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fmu = std::numeric_limits<double>::max();
        munew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(munew_mZ2minus) << endl;
        // std::cout << "New mu = " << munewGUTs_minus[6] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
        muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muminusEWSB == true) {
            muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        // if (muminusEWSB == true) {
        //     muminusEWSB = BFB_check(checkweaksols);
        // }
        muminusNoCCB = CCB_Check(checkweaksols);
        if (!(muminusEWSB) || !(muminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(mucurrentlogQSUSY), munew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                muminusNoCCB = false;
            }
        }
        if (!(muminusNoCCB)) {
            break;
        } 
        vector<double> muoldGUTs_minus = munewGUTs_minus;
        // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        while ((numStepsDone < 100) && (curr_lsq_eval > Mu_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(munewGUTs_minus, munew_mZ2minus, 6, munewlogQSUSY, munewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(munewlogQSUSY), munew_mZ2minus);
            
            double FMu = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (munew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / munew_mZ2minus));
            double GMu = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMu) - (current_derivatives[2] * FMu));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaMuNum = ((current_derivatives[3] * FMu) - (current_derivatives[1] * GMu));
            double DeltaMu = DeltaMuNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaMu, 2.0) + pow(DeltaTanb, 2.0);
            munewGUTs_minus[43] = munewGUTs_minus[43] - DeltaTanb;
            munewGUTs_minus[6] = munewGUTs_minus[6] - DeltaMu;
            munewGUTs_minus[42] = BGUT_original * munewGUTs_minus[6];
            newmuminus = munewGUTs_minus[6];
            if ((isnan(munewGUTs_minus[6])) || (isnan(munewGUTs_minus[43])) || (isnan(munewGUTs_minus[42]))) {
                muminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!muminusEWSB) {
            // std::cout << "Failed to converge" << endl;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(munewGUTs_minus, munew_mZ2minus, current_logQSUSY, current_logQGUT, 6);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            muminusEWSB = false;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
        }
        if ((munewGUTs_minus[43] < 3.0) || (munewGUTs_minus[43] > 60.0)) {
            muminusEWSB = false;
        }
    }
    // std::cout << "mu(ABDS, minus) = " << munewGUTs_minus[6] << endl; 
    double mu_GUT_minus = munewGUTs_minus[6];
    if (mu_GUT_minus == GUT_boundary_conditions[6]) {
        mu_GUT_minus = copysign(boost::math::float_prior(abs(mu_GUT_minus)), mu_GUT_minus);
    }

    lambdaMu = 0.5;
    Mu_least_Sq_Tol = 1.0e-4;
    prev_fmu = std::numeric_limits<double>::max();
    while ((muplusEWSB) && (muplusNoCCB) && (abs(munewGUTs_plus[6]) > 10.0) && ((munew_mZ2plus > (45.5938 * 45.5938)) && (munew_mZ2plus < (364.7504 * 364.7504)))) {
        lambdaMu = 0.5;
        Mu_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fmu = std::numeric_limits<double>::max();
        munew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(munew_mZ2plus) << endl;
        // std::cout << "New mu(GUT) = " << munewGUTs_plus[6] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
        muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muplusEWSB == true) {
            muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        // if (muplusEWSB == true) {
        //     muplusEWSB = BFB_check(checkweaksols);
        // }
        muplusNoCCB = CCB_Check(checkweaksols);
        if (!(muplusEWSB) || !(muplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(mucurrentlogQSUSY), munew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                muplusNoCCB = false;
            }
        }
        if (!(muplusNoCCB)) {
            break;
        } 
        vector<double> muoldGUTs_plus = munewGUTs_plus;
        // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        while ((numStepsDone < 100) && (curr_lsq_eval > Mu_least_Sq_Tol)) {
            // std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(munewGUTs_plus, munew_mZ2plus, 6, munewlogQSUSY, munewlogQGUT);

            vector<double> weaksol_plus = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(munewlogQSUSY), munew_mZ2plus);
            
            double FMu = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (munew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / munew_mZ2plus));
            double GMu = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMu) - (current_derivatives[2] * FMu));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaMuNum = ((current_derivatives[3] * FMu) - (current_derivatives[1] * GMu));
            double DeltaMu = DeltaMuNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaMu, 2.0) + pow(DeltaTanb, 2.0);
            //std::cout << curr_lsq_eval << " = current L2" << endl;

            munewGUTs_plus[43] = munewGUTs_plus[43] - DeltaTanb;
            munewGUTs_plus[6] = munewGUTs_plus[6] - DeltaMu;
            munewGUTs_plus[42] = BGUT_original * munewGUTs_plus[6];
            
            if ((isnan(munewGUTs_plus[6])) || (isnan(munewGUTs_plus[42])) || (isnan(munewGUTs_plus[43]))) {
                muplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!muplusEWSB) {
            // std::cout << "Failed to converge" << endl;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
            break;
        }   
        vector<double> check_valid_solutions = get_F_G_vals(munewGUTs_plus, munew_mZ2plus, current_logQSUSY, current_logQGUT, 6);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            // std::cout << "Failed to converge" << endl;
            muplusEWSB = false;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
        }
        if ((munewGUTs_plus[43] < 3.0) || (munewGUTs_plus[43] > 60.0)) {
            muplusEWSB = false;
        }         
    }
    // std::cout << "mu(ABDS, plus) = " << munewGUTs_plus[6] << endl; 
    double mu_GUT_plus = munewGUTs_plus[6];
    if (mu_GUT_plus == GUT_boundary_conditions[6]) {
        mu_GUT_plus = copysign(boost::math::float_next(abs(mu_GUT_plus)), mu_GUT_plus);
    }

    std::cout << "ABDS window established for mu variation." << endl;

    bool ABDSminuscheck = (muminusEWSB && muminusNoCCB); 
    bool ABDSpluscheck = (muplusEWSB && muplusNoCCB);
    double mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (mu_GUT_minus <= mu_GUT_plus) {
            mu_TOTAL_GUT_minus = boost::math::float_prior(mu_GUT_minus);
            mu_TOTAL_GUT_plus = boost::math::float_next(mu_GUT_plus);
        } else {
            mu_TOTAL_GUT_minus = boost::math::float_next(mu_GUT_minus);
            mu_TOTAL_GUT_plus = boost::math::float_prior(mu_GUT_plus);
        }

        std::cout << "General window established for mu variation." << endl;

        return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
    }

    while ((muminusEWSB) && (muminusNoCCB) && (abs(munewGUTs_minus[6]) > 10.0) && ((munew_mZ2minus > (1.0)))) {
        lambdaMu = 0.5;
        Mu_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fmu = std::numeric_limits<double>::max();
        munew_mZ2minus *= 0.99;
        int numStepsDone = 0;
        // std::cout << "New mZ = " << sqrt(munew_mZ2minus) << endl;
        // std::cout << "New mu = " << munewGUTs_minus[6] << endl;
        // std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
        muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muminusEWSB == true) {
            muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        // if (muminusEWSB == true) {
        //     muminusEWSB = BFB_check(checkweaksols);
        // }
        muminusNoCCB = CCB_Check(checkweaksols);
        if (!(muminusEWSB) || !(muminusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(mucurrentlogQSUSY), munew_mZ2minus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                muminusNoCCB = false;
            }
        }
        if (!(muminusNoCCB)) {
            break;
        } 
        vector<double> muoldGUTs_minus = munewGUTs_minus;
        // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        while ((numStepsDone < 100) && (curr_lsq_eval > Mu_least_Sq_Tol)) {
            vector<double> current_derivatives = single_var_deriv_approxes(munewGUTs_minus, munew_mZ2minus, 6, munewlogQSUSY, munewlogQGUT);
            
            vector<double> weaksol_minus = solveODEs(munewGUTs_minus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
            vector<double> RadCorrsMinus = radcorr_calc(weaksol_minus, exp(munewlogQSUSY), munew_mZ2minus);
            
            double FMu = (1.0 / 2.0) - (((weaksol_minus[26] + RadCorrsMinus[1] - ((weaksol_minus[25] + RadCorrsMinus[0]) * pow(weaksol_minus[43], 2.0))) / (munew_mZ2minus * (pow(weaksol_minus[43], 2.0) - 1.0))) - (weaksol_minus[6] * weaksol_minus[6] / munew_mZ2minus));
            double GMu = weaksol_minus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_minus[42] / (weaksol_minus[25] + RadCorrsMinus[0] + weaksol_minus[26] + RadCorrsMinus[1] + (2.0 * pow(weaksol_minus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMu) - (current_derivatives[2] * FMu));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaMuNum = ((current_derivatives[3] * FMu) - (current_derivatives[1] * GMu));
            double DeltaMu = DeltaMuNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaMu, 2.0) + pow(DeltaTanb, 2.0);
            munewGUTs_minus[43] = munewGUTs_minus[43] - DeltaTanb;
            munewGUTs_minus[6] = munewGUTs_minus[6] - DeltaMu;
            munewGUTs_minus[42] = BGUT_original * munewGUTs_minus[6];
            newmuminus = munewGUTs_minus[6];
            if ((isnan(munewGUTs_minus[6])) || (isnan(munewGUTs_minus[43])) || (isnan(munewGUTs_minus[42]))) {
                muminusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!muminusEWSB) {
            //std::cout << "Failed to converge" << endl;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
            break;
        }
        vector<double> check_valid_solutions = get_F_G_vals(munewGUTs_minus, munew_mZ2minus, current_logQSUSY, current_logQGUT, 6);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            //std::cout << "Failed to converge" << endl;
            muminusEWSB = false;
            munewGUTs_minus[6] = muoldGUTs_minus[6];
            munewGUTs_minus[42] = muoldGUTs_minus[42];
            munewGUTs_minus[43] = muoldGUTs_minus[43];
        }
        if ((munewGUTs_minus[43] < 3.0) || (munewGUTs_minus[43] > 60.0)) {
            muminusEWSB = false;
        }
    }
    //std::cout << "mu(total, minus) = " << munewGUTs_minus[6] << endl; 
    mu_TOTAL_GUT_minus = munewGUTs_minus[6];

    while ((muplusEWSB) && (muplusNoCCB) && (abs(munewGUTs_plus[6]) > 10.0) && ((munew_mZ2plus > (1.0)))) {
        lambdaMu = 0.5;
        Mu_least_Sq_Tol = 1.0e-4;
        curr_lsq_eval = std::numeric_limits<double>::max();
        prev_fmu = std::numeric_limits<double>::max();
        munew_mZ2plus *= 1.01;
        int numStepsDone = 0;
        //std::cout << "New mZ = " << sqrt(munew_mZ2plus) << endl;
        //std::cout << "New mu(GUT) = " << munewGUTs_plus[6] << endl;
        //std::cout << "---------------------------------------" << endl;
        vector<double> checkweaksols = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
        vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
        muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
        // Checking loop-level EWSB
        if (muplusEWSB == true) {
            muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
        }
        // if (muplusEWSB == true) {
        //     muplusEWSB = BFB_check(checkweaksols);
        // }
        muplusNoCCB = CCB_Check(checkweaksols);
        if (!(muplusEWSB) || !(muplusNoCCB)) {
            break;
        } 
        vector<double> tree_level_masses = TreeMassCalculator(checkweaksols, exp(mucurrentlogQSUSY), munew_mZ2plus);
        for (double value : tree_level_masses) {
            if (value < 0) {
                muplusNoCCB = false;
            }
        }
        if (!(muplusNoCCB)) {
            break;
        } 
        vector<double> muoldGUTs_plus = munewGUTs_plus;
        // Finish out convergence with Newton-Raphson now that we have a good enough initial guess
        while ((numStepsDone < 100) && (curr_lsq_eval > Mu_least_Sq_Tol)) {
            //std::cout << numStepsDone << " steps done" << endl;
            vector<double> current_derivatives = single_var_deriv_approxes(munewGUTs_plus, munew_mZ2plus, 6, munewlogQSUSY, munewlogQGUT);

            vector<double> weaksol_plus = solveODEs(munewGUTs_plus, mucurrentlogQGUT, mucurrentlogQSUSY, copysign(1.0e-6, (mucurrentlogQSUSY - mucurrentlogQGUT)));
            vector<double> RadCorrsplus = radcorr_calc(weaksol_plus, exp(munewlogQSUSY), munew_mZ2plus);
            
            double FMu = (1.0 / 2.0) - (((weaksol_plus[26] + RadCorrsplus[1] - ((weaksol_plus[25] + RadCorrsplus[0]) * pow(weaksol_plus[43], 2.0))) / (munew_mZ2plus * (pow(weaksol_plus[43], 2.0) - 1.0))) - (weaksol_plus[6] * weaksol_plus[6] / munew_mZ2plus));
            double GMu = weaksol_plus[43] - tan(0.5 * (M_PI - asin(2.0 * weaksol_plus[42] / (weaksol_plus[25] + RadCorrsplus[0] + weaksol_plus[26] + RadCorrsplus[1] + (2.0 * pow(weaksol_plus[6], 2.0))))));

            double DeltaTanbNum = ((current_derivatives[0] * GMu) - (current_derivatives[2] * FMu));
            double DeltaDenom = ((current_derivatives[0] * current_derivatives[3]) - (current_derivatives[1] * current_derivatives[2]));
            double DeltaTanb = DeltaTanbNum / DeltaDenom;

            double DeltaMuNum = ((current_derivatives[3] * FMu) - (current_derivatives[1] * GMu));
            double DeltaMu = DeltaMuNum / DeltaDenom;

            curr_lsq_eval = pow(DeltaMu, 2.0) + pow(DeltaTanb, 2.0);
            //std::cout << curr_lsq_eval << " = current L2" << endl;

            munewGUTs_plus[43] = munewGUTs_plus[43] - DeltaTanb;
            munewGUTs_plus[6] = munewGUTs_plus[6] - DeltaMu;
            munewGUTs_plus[42] = BGUT_original * munewGUTs_plus[6];
            
            if ((isnan(munewGUTs_plus[6])) || (isnan(munewGUTs_plus[42])) || (isnan(munewGUTs_plus[43]))) {
                muplusEWSB = false;
                break;
            }
            numStepsDone++;
        }
        if (!muplusEWSB) {
            //std::cout << "Failed to converge" << endl;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
            break;
        }   
        vector<double> check_valid_solutions = get_F_G_vals(munewGUTs_plus, munew_mZ2plus, current_logQSUSY, current_logQGUT, 6);
        if ((abs(check_valid_solutions[0]) > 1.0e-2) || (abs(check_valid_solutions[1]) > 1.0e-2)) {
            //std::cout << "Failed to converge" << endl;
            muplusEWSB = false;
            munewGUTs_plus[6] = muoldGUTs_plus[6];
            munewGUTs_plus[42] = muoldGUTs_plus[42];
            munewGUTs_plus[43] = muoldGUTs_plus[43];
        }
        if ((munewGUTs_plus[43] < 3.0) || (munewGUTs_plus[43] > 60.0)) {
            muplusEWSB = false;
        }         
    }
    //std::cout << "mu(total, plus) = " << munewGUTs_plus[6] << endl; 
    mu_TOTAL_GUT_plus = munewGUTs_plus[6];

    if ((abs(mu_TOTAL_GUT_minus - mu_GUT_minus) < 1.0e-12) && (abs(mu_TOTAL_GUT_plus - mu_GUT_plus) < 1.0e-12)) {
        if (mu_GUT_minus <= mu_GUT_plus) {
            mu_TOTAL_GUT_minus = boost::math::float_prior(mu_GUT_minus);
            mu_TOTAL_GUT_plus = boost::math::float_next(mu_GUT_plus);
        } else {
            mu_TOTAL_GUT_minus = boost::math::float_next(mu_GUT_minus);
            mu_TOTAL_GUT_plus = boost::math::float_prior(mu_GUT_plus);
        }

        std::cout << "General window established for mu variation." << endl;

        return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
    }

    std::cout << "General window established for mu variation." << endl;

    return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
}

double DSN_calc(int precselno, std::vector<double> GUT_boundary_conditions,
                double& current_mZ2, double& current_logQSUSY,
                double& current_logQGUT, int& nF, int& nD) {
    double DSN, DSN_soft_num, DSN_soft_denom, DSN_higgsino, newterm;
    DSN = 0.0;
    double t_target = log(500.0);
    std::cout << "This may take a while...\n\nProgress:\n-----------------------------------------------\n" << endl;
    if ((precselno == 1) || (precselno == 2)) {
        // Compute mu windows around original point
        vector<double> muinitGUTBCs = GUT_boundary_conditions;
        vector<double> muwindows = DSN_mu_windows(muinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_higgsino = abs(log10(abs(muwindows[1] / muwindows[0])));
        DSN_higgsino /= abs(muwindows[1] - muwindows[0]);
        newterm = DSN_higgsino;
        // Total normalization
        DSN_higgsino = abs(log10(abs(muwindows[3] / muwindows[2])));
        DSN_higgsino /= abs(muwindows[3] - muwindows[2]);
        if ((abs(DSN_higgsino - newterm) < numeric_limits<double>::epsilon()) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan(DSN_higgsino)) || (DSN_higgsino == 0.0) || isinf(DSN_higgsino)) {
            newterm = abs(log10(1.0 + (numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]))))\
                / abs(numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
        std::cout << "DSN after higgsino = " << DSN << endl;
        
        // Now do same thing with mHu^2(GUT)
        vector<double> mHu2initGUTBCs = GUT_boundary_conditions;
        vector<double> mHu2windows = DSN_specific_windows(mHu2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 25);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHu2windows[1])), mHu2windows[1]) - copysign(sqrt(abs(mHu2windows[0])), mHu2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHu2windows[1])), mHu2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHu2windows[0])), mHu2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHu2windows[3])), mHu2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHu2windows[2])), mHu2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHu2windows[3])), mHu2windows[3]) - copysign(sqrt(abs(mHu2windows[2])), mHu2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[25]))), boost::math::float_next(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[25]))), boost::math::float_prior(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mHd^2(GUT)
        vector<double> mHd2initGUTBCs = GUT_boundary_conditions;
        vector<double> mHd2windows = DSN_specific_windows(mHd2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 26);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHd2windows[1])), mHd2windows[1]) - copysign(sqrt(abs(mHd2windows[0])), mHd2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHd2windows[1])), mHd2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHd2windows[0])), mHd2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mHd2windows[3])), mHd2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mHd2windows[2])), mHd2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mHd2windows[3])), mHd2windows[3]) - copysign(sqrt(abs(mHd2windows[2])), mHd2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[26]))), boost::math::float_next(GUT_boundary_conditions[26])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[26]))), boost::math::float_prior(GUT_boundary_conditions[26])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[26])), GUT_boundary_conditions[26]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M1
        vector<double> M1initGUTBCs = GUT_boundary_conditions;
        vector<double> M1windows = DSN_specific_windows(M1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 3);
        DSN_soft_denom = abs(M1windows[1] - M1windows[0]);
        DSN_soft_num = soft_prob_calc(M1windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M1windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M1windows[3] - M1windows[2]);
        DSN_soft_num = soft_prob_calc(M1windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M1windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[3]) - boost::math::float_prior(GUT_boundary_conditions[3])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M2
        vector<double> M2initGUTBCs = GUT_boundary_conditions;
        vector<double> M2windows = DSN_specific_windows(M2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 4);
        DSN_soft_denom = abs(M2windows[1] - M2windows[0]);
        DSN_soft_num = soft_prob_calc(M2windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M2windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M2windows[3] - M2windows[2]);
        DSN_soft_num = soft_prob_calc(M2windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M2windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[4]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[4]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[4]) - boost::math::float_prior(GUT_boundary_conditions[4])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[4], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[4], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with M3
        vector<double> M3initGUTBCs = GUT_boundary_conditions;
        vector<double> M3windows = DSN_specific_windows(M3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 5);
        DSN_soft_denom = abs(M3windows[1] - M3windows[0]);
        DSN_soft_num = soft_prob_calc(M3windows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M3windows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(M3windows[3] - M3windows[2]);
        DSN_soft_num = soft_prob_calc(M3windows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(M3windows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[5]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[5]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[5]) - boost::math::float_prior(GUT_boundary_conditions[5])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[5], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[5], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ3
        vector<double> MQ3initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ3windows = DSN_specific_windows(MQ3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 29);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ3windows[1])), MQ3windows[1])  - copysign(sqrt(abs(MQ3windows[0])), MQ3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ3windows[1])), MQ3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ3windows[0])), MQ3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ3windows[3])), MQ3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ3windows[2])), MQ3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ3windows[3])), MQ3windows[3]) - copysign(sqrt(abs(MQ3windows[2])), MQ3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[29]))), boost::math::float_next(GUT_boundary_conditions[29])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[29]))), boost::math::float_prior(GUT_boundary_conditions[29])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[29])), GUT_boundary_conditions[29]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ2
        vector<double> MQ2initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ2windows = DSN_specific_windows(MQ2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 28);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ2windows[1])), MQ2windows[1])  - copysign(sqrt(abs(MQ2windows[0])), MQ2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ2windows[1])), MQ2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ2windows[0])), MQ2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ2windows[3])), MQ2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ2windows[2])), MQ2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ2windows[3])), MQ2windows[3]) - copysign(sqrt(abs(MQ2windows[2])), MQ2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[28]))), boost::math::float_next(GUT_boundary_conditions[28])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[28]))), boost::math::float_prior(GUT_boundary_conditions[28])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[28])), GUT_boundary_conditions[28]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mQ1
        vector<double> MQ1initGUTBCs = GUT_boundary_conditions;
        vector<double> MQ1windows = DSN_specific_windows(MQ1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 27);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ1windows[1])), MQ1windows[1])  - copysign(sqrt(abs(MQ1windows[0])), MQ1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ1windows[1])), MQ1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ1windows[0])), MQ1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(MQ1windows[3])), MQ1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(MQ1windows[2])), MQ1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(MQ1windows[3])), MQ1windows[3]) - copysign(sqrt(abs(MQ1windows[2])), MQ1windows[2]));
        std::cout << DSN_soft_num / DSN_soft_denom << endl;
        std::cout << newterm << endl;
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[27]))), boost::math::float_next(GUT_boundary_conditions[27])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[27]))), boost::math::float_prior(GUT_boundary_conditions[27])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[27])), GUT_boundary_conditions[27]), (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[27]))) - (pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[27]))));
            std::cout << DSN_soft_num / DSN_soft_denom << endl;
            std::cout << newterm << endl;
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL3
        vector<double> mL3initGUTBCs = GUT_boundary_conditions;
        vector<double> mL3windows = DSN_specific_windows(mL3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 32);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL3windows[1])), mL3windows[1])  - copysign(sqrt(abs(mL3windows[0])), mL3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL3windows[1])), mL3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL3windows[0])), mL3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL3windows[3])), mL3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL3windows[2])), mL3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL3windows[3])), mL3windows[3]) - copysign(sqrt(abs(mL3windows[2])), mL3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[32]))), boost::math::float_next(GUT_boundary_conditions[32])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[32]))), boost::math::float_prior(GUT_boundary_conditions[32])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[32])), GUT_boundary_conditions[32]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL2
        vector<double> mL2initGUTBCs = GUT_boundary_conditions;
        vector<double> mL2windows = DSN_specific_windows(mL2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 31);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL2windows[1])), mL2windows[1])  - copysign(sqrt(abs(mL2windows[0])), mL2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL2windows[1])), mL2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL2windows[0])), mL2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL2windows[3])), mL2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL2windows[2])), mL2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL2windows[3])), mL2windows[3]) - copysign(sqrt(abs(mL2windows[2])), mL2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[31]))), boost::math::float_next(GUT_boundary_conditions[31])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[31]))), boost::math::float_prior(GUT_boundary_conditions[31])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[31])), GUT_boundary_conditions[31]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mL1
        vector<double> mL1initGUTBCs = GUT_boundary_conditions;
        vector<double> mL1windows = DSN_specific_windows(mL1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 30);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL1windows[1])), mL1windows[1])  - copysign(sqrt(abs(mL1windows[0])), mL1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL1windows[1])), mL1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL1windows[0])), mL1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mL1windows[3])), mL1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mL1windows[2])), mL1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mL1windows[3])), mL1windows[3]) - copysign(sqrt(abs(mL1windows[2])), mL1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[30]))), boost::math::float_next(GUT_boundary_conditions[30])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[30]))), boost::math::float_prior(GUT_boundary_conditions[30])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[30])), GUT_boundary_conditions[30]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU3
        vector<double> mU3initGUTBCs = GUT_boundary_conditions;
        vector<double> mU3windows = DSN_specific_windows(mU3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 35);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU3windows[1])), mU3windows[1])  - copysign(sqrt(abs(mU3windows[0])), mU3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU3windows[1])), mU3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU3windows[0])), mU3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU3windows[3])), mU3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU3windows[2])), mU3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU3windows[3])), mU3windows[3]) - copysign(sqrt(abs(mU3windows[2])), mU3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[35]))), boost::math::float_next(GUT_boundary_conditions[35])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[35]))), boost::math::float_prior(GUT_boundary_conditions[35])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[35])), GUT_boundary_conditions[35]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU2
        vector<double> mU2initGUTBCs = GUT_boundary_conditions;
        vector<double> mU2windows = DSN_specific_windows(mU2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 34);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU2windows[1])), mU2windows[1])  - copysign(sqrt(abs(mU2windows[0])), mU2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU2windows[1])), mU2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU2windows[0])), mU2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU2windows[3])), mU2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU2windows[2])), mU2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU2windows[3])), mU2windows[3]) - copysign(sqrt(abs(mU2windows[2])), mU2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[34]))), boost::math::float_next(GUT_boundary_conditions[34])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[34]))), boost::math::float_prior(GUT_boundary_conditions[34])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[34])), GUT_boundary_conditions[34]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mU1
        vector<double> mU1initGUTBCs = GUT_boundary_conditions;
        vector<double> mU1windows = DSN_specific_windows(mU1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 33);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU1windows[1])), mU1windows[1])  - copysign(sqrt(abs(mU1windows[0])), mU1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU1windows[1])), mU1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU1windows[0])), mU1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mU1windows[3])), mU1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mU1windows[2])), mU1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mU1windows[3])), mU1windows[3]) - copysign(sqrt(abs(mU1windows[2])), mU1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[33]))), boost::math::float_next(GUT_boundary_conditions[33])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[33]))), boost::math::float_prior(GUT_boundary_conditions[33])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[33])), GUT_boundary_conditions[33]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD3
        vector<double> mD3initGUTBCs = GUT_boundary_conditions;
        vector<double> mD3windows = DSN_specific_windows(mD3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 38);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD3windows[1])), mD3windows[1])  - copysign(sqrt(abs(mD3windows[0])), mD3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD3windows[1])), mD3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD3windows[0])), mD3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD3windows[3])), mD3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD3windows[2])), mD3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD3windows[3])), mD3windows[3]) - copysign(sqrt(abs(mD3windows[2])), mD3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[38]))), boost::math::float_next(GUT_boundary_conditions[38])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[38]))), boost::math::float_prior(GUT_boundary_conditions[38])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[38])), GUT_boundary_conditions[38]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD2
        vector<double> mD2initGUTBCs = GUT_boundary_conditions;
        vector<double> mD2windows = DSN_specific_windows(mD2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 37);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD2windows[1])), mD2windows[1])  - copysign(sqrt(abs(mD2windows[0])), mD2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD2windows[1])), mD2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD2windows[0])), mD2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD2windows[3])), mD2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD2windows[2])), mD2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD2windows[3])), mD2windows[3]) - copysign(sqrt(abs(mD2windows[2])), mD2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[37]))), boost::math::float_next(GUT_boundary_conditions[37])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[37]))), boost::math::float_prior(GUT_boundary_conditions[37])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[37])), GUT_boundary_conditions[37]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mD1
        vector<double> mD1initGUTBCs = GUT_boundary_conditions;
        vector<double> mD1windows = DSN_specific_windows(mD1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 36);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD1windows[1])), mD1windows[1])  - copysign(sqrt(abs(mD1windows[0])), mD1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD1windows[1])), mD1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD1windows[0])), mD1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mD1windows[3])), mD1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mD1windows[2])), mD1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mD1windows[3])), mD1windows[3]) - copysign(sqrt(abs(mD1windows[2])), mD1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[36]))), boost::math::float_next(GUT_boundary_conditions[36])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[36]))), boost::math::float_prior(GUT_boundary_conditions[36])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[36])), GUT_boundary_conditions[36]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE3
        vector<double> mE3initGUTBCs = GUT_boundary_conditions;
        vector<double> mE3windows = DSN_specific_windows(mE3initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 41);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE3windows[1])), mE3windows[1])  - copysign(sqrt(abs(mE3windows[0])), mE3windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE3windows[1])), mE3windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE3windows[0])), mE3windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE3windows[3])), mE3windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE3windows[2])), mE3windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE3windows[3])), mE3windows[3]) - copysign(sqrt(abs(mE3windows[2])), mE3windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[41]))), boost::math::float_next(GUT_boundary_conditions[41])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[41]))), boost::math::float_prior(GUT_boundary_conditions[41])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[41])), GUT_boundary_conditions[41]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE2
        vector<double> mE2initGUTBCs = GUT_boundary_conditions;
        vector<double> mE2windows = DSN_specific_windows(mE2initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 40);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE2windows[1])), mE2windows[1])  - copysign(sqrt(abs(mE2windows[0])), mE2windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE2windows[1])), mE2windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE2windows[0])), mE2windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE2windows[3])), mE2windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE2windows[2])), mE2windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE2windows[3])), mE2windows[3]) - copysign(sqrt(abs(mE2windows[2])), mE2windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[40]))), boost::math::float_next(GUT_boundary_conditions[40])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[40]))), boost::math::float_prior(GUT_boundary_conditions[40])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[40])), GUT_boundary_conditions[40]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with mE1
        vector<double> mE1initGUTBCs = GUT_boundary_conditions;
        vector<double> mE1windows = DSN_specific_windows(mE1initGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 39);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE1windows[1])), mE1windows[1])  - copysign(sqrt(abs(mE1windows[0])), mE1windows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE1windows[1])), mE1windows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE1windows[0])), mE1windows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(mE1windows[3])), mE1windows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(mE1windows[2])), mE1windows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(mE1windows[3])), mE1windows[3]) - copysign(sqrt(abs(mE1windows[2])), mE1windows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[39]))), boost::math::float_next(GUT_boundary_conditions[39])), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[39]))), boost::math::float_prior(GUT_boundary_conditions[39])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[39])), GUT_boundary_conditions[39]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with at
        vector<double> atinitGUTBCs = GUT_boundary_conditions;
        vector<double> atwindows = DSN_specific_windows(atinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 16);
        DSN_soft_denom = abs(atwindows[1] - atwindows[0]);
        DSN_soft_num = soft_prob_calc(atwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(atwindows[3] - atwindows[2]);
        DSN_soft_num = soft_prob_calc(atwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[16]) - boost::math::float_prior(GUT_boundary_conditions[16])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ac
        vector<double> acinitGUTBCs = GUT_boundary_conditions;
        vector<double> acwindows = DSN_specific_windows(acinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 17);
        DSN_soft_denom = abs(acwindows[1] - acwindows[0]);
        DSN_soft_num = soft_prob_calc(acwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(acwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(acwindows[3] - acwindows[2]);
        DSN_soft_num = soft_prob_calc(acwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(acwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[17]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[17]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[17]) - boost::math::float_prior(GUT_boundary_conditions[17])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[17], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[17], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with au    
        vector<double> auinitGUTBCs = GUT_boundary_conditions;
        vector<double> auwindows = DSN_specific_windows(auinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 18);
        DSN_soft_denom = abs(auwindows[1] - auwindows[0]);
        DSN_soft_num = soft_prob_calc(auwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(auwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(auwindows[3] - auwindows[2]);
        DSN_soft_num = soft_prob_calc(auwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(auwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[18]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[18]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[18]) - boost::math::float_prior(GUT_boundary_conditions[18])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[18], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[18], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ab
        vector<double> abinitGUTBCs = GUT_boundary_conditions;
        vector<double> abwindows = DSN_specific_windows(abinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 19);
        DSN_soft_denom = abs(abwindows[1] - abwindows[0]);
        DSN_soft_num = soft_prob_calc(abwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(abwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(abwindows[3] - abwindows[2]);
        DSN_soft_num = soft_prob_calc(abwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(abwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[19]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[19]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[19]) - boost::math::float_prior(GUT_boundary_conditions[19])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[19], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[19], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with as
        vector<double> asinitGUTBCs = GUT_boundary_conditions;
        vector<double> aswindows = DSN_specific_windows(asinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 20);
        DSN_soft_denom = abs(aswindows[1] - aswindows[0]);
        DSN_soft_num = soft_prob_calc(aswindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aswindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(aswindows[3] - aswindows[2]);
        DSN_soft_num = soft_prob_calc(aswindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aswindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[20]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[20]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[20]) - boost::math::float_prior(GUT_boundary_conditions[20])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[20], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[20], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ad    
        vector<double> adinitGUTBCs = GUT_boundary_conditions;
        vector<double> adwindows = DSN_specific_windows(adinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 21);
        DSN_soft_denom = abs(adwindows[1] - adwindows[0]);
        DSN_soft_num = soft_prob_calc(adwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(adwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(adwindows[3] - adwindows[2]);
        DSN_soft_num = soft_prob_calc(adwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(adwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[21]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[21]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[21]) - boost::math::float_prior(GUT_boundary_conditions[21])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[21], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[21], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with atau
        vector<double> atauinitGUTBCs = GUT_boundary_conditions;
        vector<double> atauwindows = DSN_specific_windows(atauinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 22);
        DSN_soft_denom = abs(atauwindows[1] - atauwindows[0]);
        DSN_soft_num = soft_prob_calc(atauwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atauwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(atauwindows[3] - atauwindows[2]);
        DSN_soft_num = soft_prob_calc(atauwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(atauwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[22]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[22]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[22]) - boost::math::float_prior(GUT_boundary_conditions[22])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[22], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[22], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;
        
        // Now do same thing with amu
        vector<double> amuinitGUTBCs = GUT_boundary_conditions;
        vector<double> amuwindows = DSN_specific_windows(amuinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 23);
        DSN_soft_denom = abs(amuwindows[1] - amuwindows[0]);
        DSN_soft_num = soft_prob_calc(amuwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(amuwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(amuwindows[3] - amuwindows[2]);
        DSN_soft_num = soft_prob_calc(amuwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(amuwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[23]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[23]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[23]) - boost::math::float_prior(GUT_boundary_conditions[23])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[23], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[23], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with ae    
        vector<double> aeinitGUTBCs = GUT_boundary_conditions;
        vector<double> aewindows = DSN_specific_windows(aeinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, 24);
        DSN_soft_denom = abs(aewindows[1] - aewindows[0]);
        DSN_soft_num = soft_prob_calc(aewindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aewindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(aewindows[3] - aewindows[2]);
        DSN_soft_num = soft_prob_calc(aewindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(aewindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[24]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[24]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[24]) - boost::math::float_prior(GUT_boundary_conditions[24])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[24], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[24], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;

        // Now do same thing with B = b/mu;
        vector<double> BinitGUTBCs = GUT_boundary_conditions;
        vector<double> Bwindows = DSN_B_windows(BinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_soft_denom = abs(Bwindows[1] - Bwindows[0]);
        DSN_soft_num = soft_prob_calc(Bwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(Bwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(Bwindows[3] - Bwindows[2]);
        DSN_soft_num = soft_prob_calc(Bwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(Bwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0)
                       - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]) - boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "DSN soft term = " << abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm))) << endl;
    } else {
        // Compute mu windows around original point
        vector<double> muinitGUTBCs = GUT_boundary_conditions;
        vector<double> muwindows = DSN_mu_windows(muinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT);
        DSN_higgsino = abs(log10(abs(muwindows[1] / muwindows[0])));
        DSN_higgsino /= abs(muwindows[1] - muwindows[0]);
        newterm = DSN_higgsino;
        // Total normalization
        DSN_higgsino = abs(log10(abs(muwindows[3] / muwindows[2])));
        DSN_higgsino /= abs(muwindows[3] - muwindows[2]);
        if ((abs(DSN_higgsino - newterm) < numeric_limits<double>::epsilon()) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan(DSN_higgsino)) || (DSN_higgsino == 0.0) || isinf(DSN_higgsino)) {
            newterm = abs(log10(1.0 + (numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]))))\
                / abs(numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
        std::cout << "DSN after higgsino = " << DSN << endl;
    }    

    return DSN;
}