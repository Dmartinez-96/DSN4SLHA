#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
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

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i<threads; ++i)
            workers.emplace_back(
                [this] {
                    while(true) {
                        std::function<void()> task;

                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                [this]{ return this->stop || !this->tasks.empty(); });
                            if(this->stop && this->tasks.empty())
                                return;
                            task = std::move(this->tasks.front());
                            this->tasks.pop();
                        }

                        task();
                    }
                }
            );
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);

            // don't allow enqueueing after stopping the pool
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

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
    if (abs(B_GUT_minus - (GUT_boundary_conditions[42] / GUT_boundary_conditions[6])) < 1.0e-9) {
        B_GUT_minus = 0.9999 * B_GUT_minus;
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
    if (abs(B_GUT_plus - (GUT_boundary_conditions[42] / GUT_boundary_conditions[6])) < 1.0e-9) {
        B_GUT_plus = 1.0001 * B_GUT_plus;
    }
    Bcurr_tanbGUT = BnewGUTs_minus[43];
    Bnew_tanbGUT = Bcurr_tanbGUT;
    BcurrentlogQGUT = current_logQGUT;
    BcurrentlogQSUSY = current_logQSUSY;

    //std::cout << "ABDS window established for B variation." << endl;
    
    bool ABDSminuscheck = (BminusEWSB && BminusNoCCB); 
    bool ABDSpluscheck = (BplusEWSB && BplusNoCCB);
    double B_TOTAL_GUT_minus, B_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (abs(B_GUT_minus) <= abs(B_GUT_plus)) {
            B_TOTAL_GUT_minus = pow(10.0, -0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, 0.5) * B_GUT_plus;
        } else {
            B_TOTAL_GUT_minus = pow(10.0, 0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, -0.5) * B_GUT_plus;
        }

        //std::cout << "General window established for B variation." << endl;

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
        if (abs(B_GUT_minus) <= abs(B_GUT_plus)) {
            B_TOTAL_GUT_minus = pow(10.0, -0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, 0.5) * B_GUT_plus;
        } else {
            B_TOTAL_GUT_minus = pow(10.0, 0.5) * B_GUT_minus;
            B_TOTAL_GUT_plus = pow(10.0, -0.5) * B_GUT_plus;
        }
        //std::cout << "General window established for B variation." << endl;

        return {B_GUT_minus, B_GUT_plus, B_TOTAL_GUT_minus, B_TOTAL_GUT_plus};
    }
    //std::cout << "General window established for B variation." << endl;

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
    if (abs(pi_GUT_minus - GUT_boundary_conditions[SpecificIndex]) < 1.0e-9) {
        pi_GUT_minus = 0.9999 * pi_GUT_minus;
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
    if (abs(pi_GUT_plus - GUT_boundary_conditions[SpecificIndex]) < 1.0e-9) {
        pi_GUT_plus = 1.0001 * pi_GUT_plus;
    }

    //std::cout << "ABDS window established for " << paramName << " variation." << endl;

    bool ABDSminuscheck = (piminusEWSB && piminusNoCCB); 
    bool ABDSpluscheck = (piplusEWSB && piplusNoCCB);
    double pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (abs(pi_GUT_minus) <= abs(pi_GUT_plus)) {
            pi_TOTAL_GUT_minus = pow(10.0, -0.5) * pi_TOTAL_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, 0.5) * pi_TOTAL_GUT_plus;
        } else {
            pi_TOTAL_GUT_minus = pow(10.0, 0.5) * pi_TOTAL_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, -0.5) * pi_TOTAL_GUT_plus;
        }

        //std::cout << "General window established for " << paramName << " variation." << endl;

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
        if (abs(pi_GUT_minus) <= abs(pi_GUT_plus)) {
            pi_TOTAL_GUT_minus = pow(10.0, -0.5) * pi_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, 0.5) * pi_GUT_plus;
        } else {
            pi_TOTAL_GUT_minus = pow(10.0, 0.5) * pi_GUT_minus;
            pi_TOTAL_GUT_plus = pow(10.0, -0.5) * pi_GUT_plus;
        }

        //std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_GUT_minus, pi_GUT_plus, pi_TOTAL_GUT_minus, pi_TOTAL_GUT_plus};
    }

    //std::cout << "General window established for " << paramName << " variation." << endl;

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
    if (abs(mu_GUT_minus - GUT_boundary_conditions[6]) < 1.0e-9) {
        mu_GUT_minus = 0.9999 * mu_GUT_minus;
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
    if (abs(mu_GUT_plus - GUT_boundary_conditions[6]) < 1.0e-9) {
        mu_GUT_plus = 1.0001 * mu_GUT_minus;
    }

    //std::cout << "ABDS window established for mu variation." << endl;

    bool ABDSminuscheck = (muminusEWSB && muminusNoCCB); 
    bool ABDSpluscheck = (muplusEWSB && muplusNoCCB);
    double mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus;
    if (!(ABDSminuscheck) && !(ABDSpluscheck)) {
        if (abs(mu_GUT_minus) <= abs(mu_GUT_plus)) {
            mu_TOTAL_GUT_minus = pow(10.0, -0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = pow(10.0, 0.5) * mu_GUT_plus;
        } else {
            mu_TOTAL_GUT_minus = pow(10.0, 0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = pow(10.0, -0.5) * mu_GUT_plus;
        }

        //std::cout << "General window established for mu variation." << endl;

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
        if (abs(mu_GUT_minus) <= abs(mu_GUT_plus)) {
            mu_TOTAL_GUT_minus = pow(10.0, -0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = pow(10.0, 0.5) * mu_GUT_plus;
        } else {
            mu_TOTAL_GUT_minus = pow(10.0, 0.5) * mu_GUT_minus;
            mu_TOTAL_GUT_plus = pow(10.0, -0.5) * mu_GUT_plus;
        }

        //std::cout << "General window established for mu variation." << endl;

        return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
    }

    //std::cout << "General window established for mu variation." << endl;

    return {mu_GUT_minus, mu_GUT_plus, mu_TOTAL_GUT_minus, mu_TOTAL_GUT_plus};
}

double DSN_term(int paramselno, std::vector<double> GUT_boundary_conditions,
                double current_mZ2, double current_logQSUSY,
                double current_logQGUT) {
    int nF = 1;
    int nD = 0;
    double DSN, DSN_soft_num, DSN_soft_denom, DSN_higgsino, newterm;
    DSN = 0.0;
    double t_target = log(500.0);
    // Handle different cases
    if (paramselno == 6) {
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
            newterm = abs(log10(1.0001 / 0.9999))\
                / abs(0.0002 * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
        std::cout << "On parameter " << paramselno << " DSN = " << DSN << endl;
    } else if (paramselno == 42) {
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
            newterm = (soft_prob_calc(1.0001 * (GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(0.9999 * (GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(0.0002 * (GUT_boundary_conditions[42] / GUT_boundary_conditions[6])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((pow(10.0, 0.5) - pow(10.0, -0.5)) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6]);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "On parameter " << paramselno << " DSN = " << DSN << endl;
    } else if (paramselno >= 25 && paramselno < 42) {
        vector<double> piinitGUTBCs = GUT_boundary_conditions;
        vector<double> piwindows = DSN_specific_windows(piinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, paramselno);
        DSN_soft_denom = abs(copysign(sqrt(abs(piwindows[1])), piwindows[1]) - copysign(sqrt(abs(piwindows[0])), piwindows[0]));
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(piwindows[1])), piwindows[1]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(piwindows[0])), piwindows[0]), (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(piwindows[3])), piwindows[3]), (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(copysign(sqrt(abs(piwindows[2])), piwindows[2]), (2.0 * nF) + (1.0 * nD) - 1.0);
        DSN_soft_denom = abs(copysign(sqrt(abs(piwindows[3])), piwindows[3]) - copysign(sqrt(abs(piwindows[2])), piwindows[2]));
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(copysign(sqrt(abs(1.0001 * (GUT_boundary_conditions[paramselno]))), (GUT_boundary_conditions[paramselno])), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(copysign(sqrt(abs(0.9999 * (GUT_boundary_conditions[paramselno]))), (GUT_boundary_conditions[paramselno])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs((copysign(sqrt(abs(0.0002 * GUT_boundary_conditions[paramselno])), GUT_boundary_conditions[paramselno]))));
            DSN_soft_num = soft_prob_calc(copysign(sqrt(abs(pow(10.0, 0.5) * GUT_boundary_conditions[paramselno])), GUT_boundary_conditions[paramselno]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(sqrt(abs(pow(10.0, -0.5) * GUT_boundary_conditions[paramselno])), GUT_boundary_conditions[paramselno]), (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((copysign(sqrt((pow(10.0, 0.5) - pow(10.0, -0.5)) * abs(GUT_boundary_conditions[paramselno])), GUT_boundary_conditions[paramselno])));
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "On parameter " << paramselno << " DSN = " << DSN << endl;
    } else if (paramselno >= 3 && paramselno <= 5) {
        vector<double> piinitGUTBCs = GUT_boundary_conditions;
        vector<double> piwindows = DSN_specific_windows(piinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, paramselno);
        DSN_soft_denom = abs(piwindows[1] - piwindows[0]);
        DSN_soft_num = soft_prob_calc(piwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(piwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(piwindows[3] - piwindows[2]);
        DSN_soft_num = soft_prob_calc(piwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(piwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(1.0001 * (GUT_boundary_conditions[paramselno]), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(0.9999 * (GUT_boundary_conditions[paramselno]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs((0.0002) * (GUT_boundary_conditions[paramselno])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[paramselno], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[paramselno], (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((pow(10.0, 0.5) - pow(10.0, -0.5)) * GUT_boundary_conditions[paramselno]);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "On parameter " << paramselno << " DSN = " << DSN << endl;
    } else {
        vector<double> piinitGUTBCs = GUT_boundary_conditions;
        vector<double> piwindows = DSN_specific_windows(piinitGUTBCs, current_mZ2, current_logQSUSY, current_logQGUT, paramselno);
        DSN_soft_denom = abs(piwindows[1] - piwindows[0]);
        DSN_soft_num = soft_prob_calc(piwindows[1], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(piwindows[0], (2.0 * nF) + (1.0 * nD) - 1.0);
        newterm = DSN_soft_num / DSN_soft_denom;
        // Total normalization
        DSN_soft_denom = abs(piwindows[3] - piwindows[2]);
        DSN_soft_num = soft_prob_calc(piwindows[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
            - soft_prob_calc(piwindows[2], (2.0 * nF) + (1.0 * nD) - 1.0);
        if ((abs((DSN_soft_num / DSN_soft_denom) - newterm) < (numeric_limits<double>::epsilon())) || (isnan(newterm)) || (newterm == 0.0) || isinf(newterm) || (isnan((DSN_soft_num / DSN_soft_denom))) || ((DSN_soft_num / DSN_soft_denom) == 0.0) || isinf((DSN_soft_num / DSN_soft_denom))) {
            newterm = (soft_prob_calc(1.0001 * max(GUT_boundary_conditions[paramselno], 1.0e-6), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(0.9999 * max(GUT_boundary_conditions[paramselno], 1.0e-6), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(0.0002 * max(GUT_boundary_conditions[paramselno], 1.0e-6)));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * max(GUT_boundary_conditions[paramselno], 1.0e-6), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * max(GUT_boundary_conditions[paramselno], 1.0e-6), (2.0 * nF) + (1.0 * nD) - 1.0);
            DSN_soft_denom = abs((pow(10.0, 0.5) - pow(10.0, -0.5)) * max(GUT_boundary_conditions[paramselno], 1.0e-6));
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
        std::cout << "On parameter " << paramselno << " DSN = " << DSN << endl;
    }

    return DSN;
}

double DSN_calc(std::vector<double> GUT_boundary_conditions,
                double current_mZ2, double current_logQSUSY,
                double current_logQGUT) {
    ThreadPool pool(10);
    vector<std::future<double>> results;
    vector<int> paramselnoList = {3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
    double runningSumDSN = 0.0;
    mutex sumMutex;

    for (int paramNo : paramselnoList) {
        auto task = pool.enqueue(
            [paramNo, &GUT_boundary_conditions, current_mZ2, current_logQSUSY, current_logQGUT]() {
                return DSN_term(paramNo, GUT_boundary_conditions, current_mZ2, current_logQSUSY, current_logQGUT);
            }
        );
        results.push_back(move(task));
    }

    for (auto& future : results) {
        double taskResult = future.get();
        std::cout << "Delta_SN contribution = " << taskResult << endl;
        lock_guard<mutex> lock(sumMutex);
        runningSumDSN += taskResult;
    }

    return runningSumDSN;
}
