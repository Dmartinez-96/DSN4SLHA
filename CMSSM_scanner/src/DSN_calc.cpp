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

double signed_square(double x2, double h) {
    double signedsqrt = copysign(sqrt(abs(x2)), x2);
    double shiftedvalue = signedsqrt + h;
    double signedsquare;
    if ((signum(signedsqrt) != signum(shiftedvalue))) {
        signedsquare = (-1.0) * copysign(shiftedvalue * shiftedvalue, x2);
    } else {
        signedsquare = copysign(shiftedvalue * shiftedvalue, x2);
    }
    return signedsquare;
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

double first_derivative_calc(double hStep, double pm2h, double pmh, double pph, double pp2h) {
    return ((pm2h / 12.0) - (2.0 * pmh / 3.0) + (2.0 * pph / 3.0) - (pp2h / 12.0)) / hStep;
}

double second_derivative_calc(double hStep, double pStart, double pm2h, double pmh, double pph, double pp2h) {
    return (((-1.0) * pm2h / 12.0) + (4.0 * pmh / 3.0) - (5.0 * pStart / 2.0) + (4.0 * pph / 3.0) - (pp2h / 12.0)) / (hStep * hStep);
}

double mixed_second_derivative_calc(double pStep, double tStep, double fm2pm2t, double fm2pmt, double fm2ppt, double fm2pp2t,
                                    double fmpm2t, double fmpmt, double fmppt, double fmpp2t, double fppm2t, double fppmt, double fpppt,
                                    double fppp2t, double fp2pm2t, double fp2pmt, double fp2ppt, double fp2pp2t) {
    return ((1.0 / (32.0 * pStep * tStep))
            * ((4.0 * fpppt) - fp2ppt - fppp2t + (2.0 * fp2pp2t) - (4.0 * fmppt) + fm2ppt + fmpp2t - (2.0 * fm2pp2t)
               - (4.0 * fppmt) + fp2pmt + fppm2t - (2.0 * fp2pm2t) + (4.0 * fmpmt) - fm2pmt - fmpm2t + (2.0 * fm2pm2t)));
}

double calculate_approx_mZ2(vector<double> weak_solutions, double explogQSUSY, double mZ2Value) {
    vector<double> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return 2.0 * ((((weak_solutions[26] + calculateRadCorrs[1] - ((weak_solutions[25] + calculateRadCorrs[0]) * weak_solutions[43] * weak_solutions[43]))) / ((weak_solutions[43] * weak_solutions[43]) - 1.0)) - (weak_solutions[6] * weak_solutions[6]));
}

double calculate_approx_tanb(vector<double> weak_solutions, double explogQSUSY, double mZ2Value) {
    vector<double> calculateRadCorrs = radcorr_calc(weak_solutions, explogQSUSY, mZ2Value);
    return tan(0.5 * (M_PI - asin(abs(2.0 * weak_solutions[42] / (weak_solutions[25] + weak_solutions[26] + calculateRadCorrs[0] + calculateRadCorrs[1] + (2.0 * weak_solutions[6] * weak_solutions[6]))))));
}

vector<double> single_var_deriv_approxes(vector<double>& original_weak_conditions, double& fixed_mZ2_val, int idx_to_shift, double& logQSUSYval) {
    double p_orig, h_p, p_plus, p_minus, p_plusplus, p_minusminus;
    if (idx_to_shift == 42) {
        p_orig = original_weak_conditions[idx_to_shift] / original_weak_conditions[6];
        h_p = min(0.95, max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), (1.0 / 5.0)), 1.0e-9));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    } else if ((idx_to_shift >= 25) && (idx_to_shift <= 41)) {
        p_orig = sqrt(abs(original_weak_conditions[idx_to_shift]));
        h_p = min(0.95, max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), (1.0 / 5.0)), 1.0e-9));
        p_plus = copysign(pow(p_orig + h_p, 2.0), original_weak_conditions[idx_to_shift]);
        p_minus = copysign(pow(p_orig - h_p, 2.0), original_weak_conditions[idx_to_shift]);
        p_plusplus = copysign(pow(p_plus + h_p, 2.0), original_weak_conditions[idx_to_shift]);
        p_minusminus = copysign(pow(p_minus - h_p, 2.0), original_weak_conditions[idx_to_shift]);
    }
    else {
        p_orig = original_weak_conditions[idx_to_shift];
        h_p = min(0.95, max(pow(10.25 * (boost::math::float_next(abs(p_orig)) - abs(p_orig)), (1.0 / 5.0)), 1.0e-9));
        p_plus = p_orig + h_p;
        p_minus = p_orig - h_p;
        p_plusplus = p_plus + h_p;
        p_minusminus = p_minus - h_p;
    }

    vector<double> newmZ2weak_plus = original_weak_conditions;
    vector<double> newmZ2weak_plusplus = original_weak_conditions;
    vector<double> newtanbweak_plus = original_weak_conditions;
    vector<double> newtanbweak_plusplus = original_weak_conditions;
    vector<double> newmZ2weak_minus = original_weak_conditions;
    vector<double> newmZ2weak_minusminus = original_weak_conditions;
    vector<double> newtanbweak_minus = original_weak_conditions;
    vector<double> newtanbweak_minusminus = original_weak_conditions;
    vector<double> newtanb_plus_p_plus_weak = original_weak_conditions;
    vector<double> newtanb_plusplus_p_plus_weak = original_weak_conditions;
    vector<double> newtanb_plus_p_plusplus_weak = original_weak_conditions;
    vector<double> newtanb_plusplus_p_plusplus_weak = original_weak_conditions;
    vector<double> newtanb_plus_p_minus_weak = original_weak_conditions;
    vector<double> newtanb_plusplus_p_minus_weak = original_weak_conditions;
    vector<double> newtanb_plus_p_minusminus_weak = original_weak_conditions;
    vector<double> newtanb_plusplus_p_minusminus_weak = original_weak_conditions;
    vector<double> newtanb_minus_p_plus_weak = original_weak_conditions;
    vector<double> newtanb_minusminus_p_plus_weak = original_weak_conditions;
    vector<double> newtanb_minus_p_plusplus_weak = original_weak_conditions;
    vector<double> newtanb_minusminus_p_plusplus_weak = original_weak_conditions;
    vector<double> newtanb_minus_p_minus_weak = original_weak_conditions;
    vector<double> newtanb_minusminus_p_minus_weak = original_weak_conditions;
    vector<double> newtanb_minus_p_minusminus_weak = original_weak_conditions;
    vector<double> newtanb_minusminus_p_minusminus_weak = original_weak_conditions;

    double tanb_orig = original_weak_conditions[43];
    double h_tanb = pow(10.25 * (boost::math::float_next(abs(tanb_orig)) - abs(tanb_orig)), (1.0 / 5.0));
    
    newtanbweak_plus[43] = tanb_orig + h_tanb;
    newtanbweak_plusplus[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plus_p_plus_weak[43] = tanb_orig + h_tanb;
    newtanb_plus_p_plusplus_weak[43] = tanb_orig + h_tanb;
    newtanb_plusplus_p_plus_weak[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plusplus_p_plusplus_weak[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plus_p_minus_weak[43] = tanb_orig + h_tanb;
    newtanb_plus_p_minusminus_weak[43] = tanb_orig + h_tanb;
    newtanb_plusplus_p_minus_weak[43] = tanb_orig + (2.0 * h_tanb);
    newtanb_plusplus_p_minusminus_weak[43] = tanb_orig + (2.0 * h_tanb);
    newtanbweak_minus[43] = tanb_orig - h_tanb;
    newtanbweak_minusminus[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minus_p_plus_weak[43] = tanb_orig - h_tanb;
    newtanb_minus_p_plusplus_weak[43] = tanb_orig - h_tanb;
    newtanb_minusminus_p_plus_weak[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minusminus_p_plusplus_weak[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minus_p_minus_weak[43] = tanb_orig - h_tanb;
    newtanb_minus_p_minusminus_weak[43] = tanb_orig - h_tanb;
    newtanb_minusminus_p_minus_weak[43] = tanb_orig - (2.0 * h_tanb);
    newtanb_minusminus_p_minusminus_weak[43] = tanb_orig - (2.0 * h_tanb);

    // Adjust Yukawas at Q=mt=173.2 GeV for shifted tanb points
    double wk_tanb = original_weak_conditions[43];
    vector<double> weaksols_original = original_weak_conditions;
    vector<double> weaksolstanb_plus = newtanbweak_plus;
    vector<double> weaksolstanb_minus = newtanbweak_minus;
    vector<double> weaksolstanb_plusplus = newtanbweak_plusplus;
    vector<double> weaksolstanb_minusminus = newtanbweak_minusminus;
    for (int UpYukawaIndex = 7; UpYukawaIndex < 10; ++UpYukawaIndex) {
        weaksolstanb_plus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_plus[43]));
        weaksolstanb_plusplus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_plusplus[43]));
        weaksolstanb_minus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_minus[43]));
        weaksolstanb_minusminus[UpYukawaIndex] *= sin(atan(weaksols_original[43])) / sin(atan(weaksolstanb_minusminus[43]));
    }
    for (int DownYukawaIndex = 10; DownYukawaIndex < 16; ++DownYukawaIndex) {
        weaksolstanb_plus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_plus[43]));
        weaksolstanb_plusplus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_plusplus[43]));
        weaksolstanb_minus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_minus[43]));
        weaksolstanb_minusminus[DownYukawaIndex] *= cos(atan(weaksols_original[43])) / cos(atan(weaksolstanb_minusminus[43]));
    }
    for (int YukawaIndex = 7; YukawaIndex < 16; ++YukawaIndex) {
        newtanbweak_plus[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanbweak_plusplus[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanb_plus_p_plus_weak[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanb_plus_p_plusplus_weak[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanb_plusplus_p_plus_weak[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanb_plusplus_p_plusplus_weak[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanb_plus_p_minus_weak[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanb_plus_p_minusminus_weak[YukawaIndex] = weaksolstanb_plus[YukawaIndex];
        newtanb_plusplus_p_minus_weak[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanb_plusplus_p_minusminus_weak[YukawaIndex] = weaksolstanb_plusplus[YukawaIndex];
        newtanbweak_minus[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanbweak_minusminus[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
        newtanb_minus_p_plus_weak[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanb_minus_p_plusplus_weak[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanb_minusminus_p_plus_weak[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
        newtanb_minusminus_p_plusplus_weak[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
        newtanb_minus_p_minus_weak[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanb_minus_p_minusminus_weak[YukawaIndex] = weaksolstanb_minus[YukawaIndex];
        newtanb_minusminus_p_minus_weak[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
        newtanb_minusminus_p_minusminus_weak[YukawaIndex] = weaksolstanb_minusminus[YukawaIndex];
    }
    double mZ2_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus, exp(logQSUSYval), fixed_mZ2_val);

    if (idx_to_shift == 6) {
        newmZ2weak_plus[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newmZ2weak_plusplus[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newtanb_plus_p_plus_weak[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newtanb_plusplus_p_plus_weak[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newtanb_plus_p_plusplus_weak[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newtanb_plusplus_p_plusplus_weak[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newtanb_minus_p_plus_weak[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newtanb_minusminus_p_plus_weak[42] = original_weak_conditions[42] * p_plus / original_weak_conditions[6];
        newtanb_minus_p_plusplus_weak[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newtanb_minusminus_p_plusplus_weak[42] = original_weak_conditions[42] * p_plusplus / original_weak_conditions[6];
        newmZ2weak_minus[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newmZ2weak_minusminus[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newtanb_plus_p_minus_weak[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newtanb_plusplus_p_minus_weak[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newtanb_plusplus_p_minusminus_weak[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newtanb_plus_p_minusminus_weak[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newtanb_minus_p_minus_weak[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newtanb_minusminus_p_minus_weak[42] = original_weak_conditions[42] * p_minus / original_weak_conditions[6];
        newtanb_minusminus_p_minusminus_weak[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newtanb_minus_p_minusminus_weak[42] = original_weak_conditions[42] * p_minusminus / original_weak_conditions[6];
        newmZ2weak_plus[6] = p_plus;
        newmZ2weak_plusplus[6] = p_plusplus;
        newtanb_plus_p_plus_weak[6] = p_plus;
        newtanb_plusplus_p_plus_weak[6] = p_plus;
        newtanb_plusplus_p_plusplus_weak[6] = p_plusplus;
        newtanb_plus_p_plusplus_weak[6] = p_plusplus;
        newtanb_minus_p_plus_weak[6] = p_plus;
        newtanb_minusminus_p_plus_weak[6] = p_plus;
        newtanb_minusminus_p_plusplus_weak[6] = p_plusplus;
        newtanb_minus_p_plusplus_weak[6] = p_plusplus;
        newmZ2weak_minus[6] = p_minus;
        newmZ2weak_minusminus[6] = p_minusminus;
        newtanb_plus_p_minus_weak[6] = p_minus;
        newtanb_plusplus_p_minus_weak[6] = p_minus;
        newtanb_plusplus_p_minusminus_weak[6] = p_minusminus;
        newtanb_plus_p_minusminus_weak[6] = p_minusminus;
        newtanb_minus_p_minus_weak[6] = p_minus;
        newtanb_minusminus_p_minus_weak[6] = p_minus;
        newtanb_minusminus_p_minusminus_weak[6] = p_minusminus;
        newtanb_minus_p_minusminus_weak[6] = p_minusminus;
    } else if (idx_to_shift == 42) {
        newmZ2weak_plus[42] = original_weak_conditions[6] * p_plus;
        newmZ2weak_plusplus[42] = original_weak_conditions[6] * p_plusplus;
        newtanb_plus_p_plus_weak[42] = original_weak_conditions[6] * p_plus;
        newtanb_plusplus_p_plus_weak[42] = original_weak_conditions[6] * p_plus;
        newtanb_plus_p_plusplus_weak[42] = original_weak_conditions[6] * p_plusplus;
        newtanb_plusplus_p_plusplus_weak[42] = original_weak_conditions[6] * p_plusplus;
        newtanb_minus_p_plus_weak[42] = original_weak_conditions[6] * p_plus;
        newtanb_minusminus_p_plus_weak[42] = original_weak_conditions[6] * p_plus;
        newtanb_minus_p_plusplus_weak[42] = original_weak_conditions[6] * p_plusplus;
        newtanb_minusminus_p_plusplus_weak[42] = original_weak_conditions[6] * p_plusplus;
        newmZ2weak_minus[42] = original_weak_conditions[6] * p_minus;
        newmZ2weak_minusminus[42] = original_weak_conditions[6] * p_minusminus;
        newtanb_plus_p_minus_weak[42] = original_weak_conditions[6] * p_minus;
        newtanb_plusplus_p_minus_weak[42] = original_weak_conditions[6] * p_minus;
        newtanb_plus_p_minusminus_weak[42] = original_weak_conditions[6] * p_minusminus;
        newtanb_plusplus_p_minusminus_weak[42] = original_weak_conditions[6] * p_minusminus;
        newtanb_minus_p_minus_weak[42] = original_weak_conditions[6] * p_minus;
        newtanb_minusminus_p_minus_weak[42] = original_weak_conditions[6] * p_minus;
        newtanb_minus_p_minusminus_weak[42] = original_weak_conditions[6] * p_minusminus;
        newtanb_minusminus_p_minusminus_weak[42] = original_weak_conditions[6] * p_minusminus;
    } else {
        newmZ2weak_plus[idx_to_shift] = p_plus;
        newmZ2weak_plusplus[idx_to_shift] = p_plusplus;
        newtanb_plus_p_plus_weak[idx_to_shift] = p_plus;
        newtanb_plusplus_p_plus_weak[idx_to_shift] = p_plus;
        newtanb_plus_p_plusplus_weak[idx_to_shift] = p_plusplus;
        newtanb_plusplus_p_plusplus_weak[idx_to_shift] = p_plusplus;
        newtanb_minus_p_plus_weak[idx_to_shift] = p_plus;
        newtanb_minusminus_p_plus_weak[idx_to_shift] = p_plus;
        newtanb_minusminus_p_plusplus_weak[idx_to_shift] = p_plusplus;
        newtanb_minus_p_plusplus_weak[idx_to_shift] = p_plusplus;
        newmZ2weak_minus[idx_to_shift] = p_minus;
        newmZ2weak_minusminus[idx_to_shift] = p_minusminus;
        newtanb_plus_p_minus_weak[idx_to_shift] = p_minus;
        newtanb_plusplus_p_minus_weak[idx_to_shift] = p_minus;
        newtanb_plus_p_minusminus_weak[idx_to_shift] = p_minusminus;
        newtanb_plusplus_p_minusminus_weak[idx_to_shift] = p_minusminus;
        newtanb_minus_p_minus_weak[idx_to_shift] = p_minus;
        newtanb_minusminus_p_minus_weak[idx_to_shift] = p_minus;
        newtanb_minus_p_minusminus_weak[idx_to_shift] = p_minusminus;
        newtanb_minusminus_p_minusminus_weak[idx_to_shift] = p_minusminus;
    }

    vector<double> weaksolsp_plus = newmZ2weak_plus;
    vector<double> weaksolsp_plusplus = newmZ2weak_plusplus;
    vector<double> weaksolstanb_plus_p_plus = newtanb_plus_p_plus_weak;
    vector<double> weaksolstanb_plusplus_p_plus = newtanb_plusplus_p_plus_weak;
    vector<double> weaksolstanb_plus_p_plusplus = newtanb_plus_p_plusplus_weak;
    vector<double> weaksolstanb_plusplus_p_plusplus = newtanb_plusplus_p_plusplus_weak;
    vector<double> weaksolstanb_minus_p_plus = newtanb_minus_p_plus_weak;
    vector<double> weaksolstanb_minusminus_p_plus = newtanb_minusminus_p_plus_weak;
    vector<double> weaksolstanb_minus_p_plusplus = newtanb_minus_p_plusplus_weak;
    vector<double> weaksolstanb_minusminus_p_plusplus = newtanb_minusminus_p_plusplus_weak;
    vector<double> weaksolsp_minus = newmZ2weak_minus;
    vector<double> weaksolsp_minusminus = newmZ2weak_minusminus;
    vector<double> weaksolstanb_plus_p_minus = newtanb_plus_p_minus_weak;
    vector<double> weaksolstanb_plusplus_p_minus = newtanb_plusplus_p_minus_weak;
    vector<double> weaksolstanb_plus_p_minusminus = newtanb_plus_p_minusminus_weak;
    vector<double> weaksolstanb_plusplus_p_minusminus = newtanb_plusplus_p_minusminus_weak;
    vector<double> weaksolstanb_minus_p_minus = newtanb_minus_p_minus_weak;
    vector<double> weaksolstanb_minusminus_p_minus = newtanb_minusminus_p_minus_weak;
    vector<double> weaksolstanb_minus_p_minusminus = newtanb_minus_p_minusminus_weak;
    vector<double> weaksolstanb_minusminus_p_minusminus = newtanb_minusminus_p_minusminus_weak;
        
    double mZ2_p_plus = calculate_approx_mZ2(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus = calculate_approx_mZ2(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_plus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_plusplus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    double mZ2_p_minus = calculate_approx_mZ2(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus = calculate_approx_mZ2(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_plus = calculate_approx_mZ2(weaksolstanb_plus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_plusplus = calculate_approx_mZ2(weaksolstanb_plusplus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_minus = calculate_approx_mZ2(weaksolstanb_minus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_minus, exp(logQSUSYval), fixed_mZ2_val);
    double mZ2_p_minusminus_tanb_minusminus = calculate_approx_mZ2(weaksolstanb_minusminus_p_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    double tanb_p_plus = calculate_approx_tanb(weaksolsp_plus, exp(logQSUSYval), fixed_mZ2_val);
    double tanb_p_plusplus = calculate_approx_tanb(weaksolsp_plusplus, exp(logQSUSYval), fixed_mZ2_val);
    
    double tanb_p_minus = calculate_approx_tanb(weaksolsp_minus, exp(logQSUSYval), fixed_mZ2_val);
    double tanb_p_minusminus = calculate_approx_tanb(weaksolsp_minusminus, exp(logQSUSYval), fixed_mZ2_val);
    
    /* Order of derivatives:
        (0: dt/dp,
         1: d^2t/dp^2,
         2: dm/dt, 
         3: dm/dp,
         4: d^2m/dt^2,
         5: d^2m/dtdp,
         6: d^2m/dp^2)
    */
    vector<double> evaluated_derivs = {first_derivative_calc(h_p, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       second_derivative_calc(h_p, wk_tanb, tanb_p_minusminus, tanb_p_minus, tanb_p_plus, tanb_p_plusplus),
                                       first_derivative_calc(h_tanb, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                       first_derivative_calc(h_p, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus),
                                       second_derivative_calc(h_tanb, fixed_mZ2_val, mZ2_tanb_minusminus, mZ2_tanb_minus, mZ2_tanb_plus, mZ2_tanb_plusplus),
                                       mixed_second_derivative_calc(h_p, h_tanb, mZ2_p_minusminus_tanb_minusminus, mZ2_p_minusminus_tanb_minus, mZ2_p_minusminus_tanb_plus, mZ2_p_minusminus_tanb_plusplus,
                                                                    mZ2_p_minus_tanb_minusminus, mZ2_p_minus_tanb_minus, mZ2_p_minus_tanb_plus, mZ2_p_minus_tanb_plusplus, mZ2_p_plus_tanb_minusminus, mZ2_p_plus_tanb_minus,
                                                                    mZ2_p_plus_tanb_plus, mZ2_p_plus_tanb_plusplus, mZ2_p_plusplus_tanb_minusminus, mZ2_p_plusplus_tanb_minus, mZ2_p_plusplus_tanb_plus, mZ2_p_plusplus_tanb_plusplus),
                                       second_derivative_calc(h_p, fixed_mZ2_val, mZ2_p_minusminus, mZ2_p_minus, mZ2_p_plus, mZ2_p_plusplus)};
    return evaluated_derivs;
}

vector<double> DSN_B_windows(vector<double> Wk_boundary_conditions, double& current_mZ2, double& current_logQSUSY) {
    vector<double> Bnewweaks_plus = Wk_boundary_conditions;
    vector<double> Bnewweaks_minus = Wk_boundary_conditions;
    double BcurrentlogQSUSY = current_logQSUSY;
    double BnewlogQSUSY = current_logQSUSY;
    double Bnew_mZ2plus = current_mZ2;
    double Bnew_mZ2minus = current_mZ2;
    bool BminusNoCCB = true;
    bool BminusEWSB = true;
    bool BplusNoCCB = true;
    bool BplusEWSB = true;

    double Bplus = Bnewweaks_plus[42] / Wk_boundary_conditions[6];
    double newBplus = Bplus;
    double tanbplus = Bnewweaks_plus[43];
    double newtanbplus = tanbplus;

    double Bminus = Bnewweaks_minus[42] / Wk_boundary_conditions[6];
    double newBminus = Bminus;
    double tanbminus = Bnewweaks_minus[43];
    double newtanbminus = tanbminus;
    double muGUT_original = Wk_boundary_conditions[6];

    // First compute width of ABDS window
    double lambdaB = 0.5;
    double B_least_Sq_Tol = 1.0e-2;
    double prev_fB = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    try {
        vector<double> current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);
        for (double deriv_value : current_derivatives) {
            if (isnan(deriv_value) || isinf(deriv_value)) {
                BminusEWSB = false;
            }
        }
        double Bstepplus, Bstepminus, bigBstep;
        Bstepminus = (boost::math::float_prior((Bnewweaks_minus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_minus[42] / Wk_boundary_conditions[6]));
            
        double mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus))
                            + (0.5 * Bstepminus * Bstepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus));
        }
        double tanbshift_minus = (current_derivatives[0] * Bstepminus) + (0.5 * current_derivatives[1] * Bstepminus * Bstepminus);
        
        bool too_sensitive_flag_minus = false, too_sensitive_flag_plus = false;
        double B_weak_minus, B_weak_plus;
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
            B_weak_minus = (Bnewweaks_minus[42] / Wk_boundary_conditions[6]) + Bstepminus;
        } 
        while ((!too_sensitive_flag_minus) && (BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > (45.5938 * 45.5938)) && (Bnew_mZ2minus < (364.7504 * 364.7504)))) {
            bigBstep = abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * Bstepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = Bnewweaks_minus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
            BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (BminusEWSB == true) {
                BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
            }
            BminusNoCCB = CCB_Check(checkweaksols);
            if (!(BminusEWSB) || !(BminusNoCCB)) {
                break;
            } 
            if (!(BminusNoCCB)) {
                break;
            } 
            vector<double> Boldweaks_minus = Bnewweaks_minus;
            Bnewweaks_minus[42] = ((Bnewweaks_minus[42] / Wk_boundary_conditions[6]) + bigBstep) * Wk_boundary_conditions[6];
                    
            if (!(BminusEWSB)) {
                Bnewweaks_minus[42] = Boldweaks_minus[42];
                break;
            }
            Bnew_mZ2minus += abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * copysign(1.0, mZ2shift_minus);
            Bnewweaks_minus[43] += abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((Bnewweaks_minus[43] < 3.0) || (Bnewweaks_minus[43] > 60.0)) {
                BminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_minus[YukIndx] *= sin(atan(Boldweaks_minus[43])) / sin(atan(Bnewweaks_minus[43]));
                    } else {
                        Bnewweaks_minus[YukIndx] *= cos(atan(Boldweaks_minus[43])) / cos(atan(Bnewweaks_minus[43]));
                    }
                }
            }        

            if (!BminusEWSB) {
                Bnewweaks_minus[42] = Boldweaks_minus[42];
                Bnewweaks_minus[43] = Boldweaks_minus[43];
                break;
            }
            Bstepminus = (boost::math::float_prior((Bnewweaks_minus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_minus[42] / Wk_boundary_conditions[6]));
            current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus))
                            + (0.5 * Bstepminus * Bstepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > 1.0) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus));
            }
            tanbshift_minus = (current_derivatives[0] * Bstepminus) + (0.5 * current_derivatives[1] * Bstepminus * Bstepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        
        B_weak_minus = Bnewweaks_minus[42] / Wk_boundary_conditions[6];
        Bstepplus = (boost::math::float_next((Bnewweaks_plus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_plus[42] / Wk_boundary_conditions[6]));
        current_derivatives = single_var_deriv_approxes(Bnewweaks_plus, Bnew_mZ2plus, 42, BnewlogQSUSY);
        
        double mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus))
                            + (0.5 * Bstepplus * Bstepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        double tanbshift_plus = (current_derivatives[0] * Bstepplus) + (0.5 * current_derivatives[1] * Bstepplus * Bstepplus);

        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus));
        }
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
            B_weak_plus = (Bnewweaks_plus[42] / Bnewweaks_plus[6]) + (Bstepplus);
        } 
        while ((!too_sensitive_flag_plus) && (BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > (45.5938 * 45.5938)) && (Bnew_mZ2plus < (364.7504 * 364.7504)))) {
            bigBstep = abs(((0.2 * sqrt(abs(Bnew_mZ2plus))) + 0.01)) * Bstepplus / abs(mZ2shift_plus);
            vector<double> checkweaksols = Bnewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
            BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (BplusEWSB == true) {
                BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
            }
            BplusNoCCB = CCB_Check(checkweaksols);
            if (!(BplusEWSB) || !(BplusNoCCB)) {
                break;
            } 
            if (!(BplusNoCCB)) {
                break;
            } 
            vector<double> Boldweaks_plus = Bnewweaks_plus;
            Bnewweaks_plus[42] = ((Bnewweaks_plus[42] / Wk_boundary_conditions[6]) + bigBstep) * Wk_boundary_conditions[6];
                    
            if (!(BplusEWSB)) {
                Bnewweaks_plus[42] = Boldweaks_plus[42];
                break;
            }
            Bnew_mZ2plus += abs(((0.2 * sqrt(abs(Bnew_mZ2plus))) + 0.01)) * copysign(1.0, (-1.0) * (Bnew_mZ2minus - (91.1876 * 91.1876)));
            Bnewweaks_plus[43] += (tanbshift_plus * ((2.0 * sqrt(Bnew_mZ2plus)) + 1.0) / abs(mZ2shift_plus));
            // Now adjust Yukawas for next iteration.
            if ((Bnewweaks_plus[43] < 3.0) || (Bnewweaks_plus[43] > 60.0)) {
                BplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_plus[YukIndx] *= sin(atan(Boldweaks_plus[43])) / sin(atan(Bnewweaks_plus[43]));
                    } else {
                        Bnewweaks_plus[YukIndx] *= cos(atan(Boldweaks_plus[43])) / cos(atan(Bnewweaks_plus[43]));
                    }
                }
            }        

            if (!BplusEWSB) {
                Bnewweaks_plus[42] = Boldweaks_plus[42];
                Bnewweaks_plus[43] = Boldweaks_plus[43];
                break;
            }
            Bstepplus = (boost::math::float_next((Bnewweaks_plus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_plus[42] / Wk_boundary_conditions[6]));
            current_derivatives = single_var_deriv_approxes(Bnewweaks_plus, Bnew_mZ2plus, 42, BnewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus))
                            + (0.5 * Bstepplus * Bstepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > 1.0) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus));
            }
            tanbshift_plus = (current_derivatives[0] * Bstepplus) + (0.5 * current_derivatives[1] * Bstepplus * Bstepplus);
            
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        } 
        
        B_weak_plus = Bnewweaks_plus[42] / Wk_boundary_conditions[6];
        
        std::cout << "ABDS window established for B variation." << endl;

        double B_TOTAL_weak_minus, B_TOTAL_weak_plus;
        
        Bstepminus = (boost::math::float_prior((Bnewweaks_minus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_minus[42] / Wk_boundary_conditions[6]));
        current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);

        mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus))
                            + (0.5 * Bstepminus * Bstepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus));
        }
        tanbshift_minus = (current_derivatives[0] * Bstepminus) + (0.5 * current_derivatives[1] * Bstepminus * Bstepminus);
        
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
        } 

        while ((!too_sensitive_flag_minus) && (BminusEWSB) && (BminusNoCCB) && ((Bnew_mZ2minus > 1.0)) && (Bnew_mZ2plus < 1.0e12)) {
            bigBstep = abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * Bstepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = Bnewweaks_minus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2minus);
            BminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (BminusEWSB == true) {
                BminusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
            }
            BminusNoCCB = CCB_Check(checkweaksols);
            if (!(BminusEWSB) || !(BminusNoCCB)) {
                break;
            } 
            if (!(BminusNoCCB)) {
                break;
            } 
            vector<double> Boldweaks_minus = Bnewweaks_minus;
            Bnewweaks_minus[42] = ((Bnewweaks_minus[42] / Wk_boundary_conditions[6]) + bigBstep) * Wk_boundary_conditions[6];
                    
            if (!(BminusEWSB)) {
                Bnewweaks_minus[42] = Boldweaks_minus[42];
                break;
            }
            Bnew_mZ2minus += abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * copysign(1.0, mZ2shift_minus);
            Bnewweaks_minus[43] += abs(((0.2 * sqrt(abs(Bnew_mZ2minus))) + 0.01)) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((Bnewweaks_minus[43] < 3.0) || (Bnewweaks_minus[43] > 60.0)) {
                BminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_minus[YukIndx] *= sin(atan(Boldweaks_minus[43])) / sin(atan(Bnewweaks_minus[43]));
                    } else {
                        Bnewweaks_minus[YukIndx] *= cos(atan(Boldweaks_minus[43])) / cos(atan(Bnewweaks_minus[43]));
                    }
                }
            }        

            if (!BminusEWSB) {
                Bnewweaks_minus[42] = Boldweaks_minus[42];
                Bnewweaks_minus[43] = Boldweaks_minus[43];
                break;
            }
            Bstepminus = (boost::math::float_prior((Bnewweaks_minus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_minus[42] / Wk_boundary_conditions[6]));
            current_derivatives = single_var_deriv_approxes(Bnewweaks_minus, Bnew_mZ2minus, 42, BnewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus))
                            + (0.5 * Bstepminus * Bstepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > 1.0) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepminus));
            }
            tanbshift_minus = (current_derivatives[0] * Bstepminus) + (0.5 * current_derivatives[1] * Bstepminus * Bstepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        B_TOTAL_weak_minus = Bnewweaks_minus[42] / Wk_boundary_conditions[6];

        Bstepplus = (boost::math::float_next((Bnewweaks_plus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_plus[42] / Wk_boundary_conditions[6]));
        current_derivatives = single_var_deriv_approxes(Bnewweaks_plus, Bnew_mZ2plus, 42, BnewlogQSUSY);

        mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus))
                            + (0.5 * Bstepplus * Bstepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus));
        }
        tanbshift_plus = (current_derivatives[0] * Bstepplus) + (0.5 * current_derivatives[1] * Bstepplus * Bstepplus);
        
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
            B_TOTAL_weak_plus = Bnewweaks_plus[6] + (Bstepplus / abs(mZ2shift_plus));
        } 

        while ((!too_sensitive_flag_plus) && (BplusEWSB) && (BplusNoCCB) && ((Bnew_mZ2plus > 1.0)) && (Bnew_mZ2plus < 1.0e12)) {
            bigBstep = abs(((0.2 * sqrt(abs(Bnew_mZ2plus))) + 0.01)) * Bstepplus / abs(mZ2shift_plus);
            
            vector<double> checkweaksols = Bnewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(BnewlogQSUSY), Bnew_mZ2plus);
            BplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (BplusEWSB == true) {
                BplusEWSB = Hessian_check(checkweaksols, exp(BnewlogQSUSY));
            }
            BplusNoCCB = CCB_Check(checkweaksols);
            if (!(BplusEWSB) || !(BplusNoCCB)) {
                break;
            } 
            if (!(BplusNoCCB)) {
                break;
            } 
            vector<double> Boldweaks_plus = Bnewweaks_plus;
            Bnewweaks_plus[42] = ((Bnewweaks_plus[42] / Wk_boundary_conditions[6]) + bigBstep) * Wk_boundary_conditions[6];
                    
            if (!(BplusEWSB)) {
                Bnewweaks_plus[42] = Boldweaks_plus[42];
                break;
            }
            Bnew_mZ2plus += abs(((0.2 * sqrt(abs(Bnew_mZ2plus))) + 0.01)) * copysign(1.0, (-1.0) * (Bnew_mZ2minus - (91.1876 * 91.1876)));
            Bnewweaks_plus[43] += (tanbshift_plus * ((2.0 * sqrt(Bnew_mZ2plus)) + 1.0) / abs(mZ2shift_plus));
            // Now adjust Yukawas for next iteration.
            if ((Bnewweaks_plus[43] < 3.0) || (Bnewweaks_plus[43] > 60.0)) {
                BplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        Bnewweaks_plus[YukIndx] *= sin(atan(Boldweaks_plus[43])) / sin(atan(Bnewweaks_plus[43]));
                    } else {
                        Bnewweaks_plus[YukIndx] *= cos(atan(Boldweaks_plus[43])) / cos(atan(Bnewweaks_plus[43]));
                    }
                }
            }        

            if (!BplusEWSB) {
                Bnewweaks_plus[42] = Boldweaks_plus[42];
                Bnewweaks_plus[43] = Boldweaks_plus[43];
                break;
            }
            Bstepplus = (boost::math::float_next((Bnewweaks_plus[42] / Wk_boundary_conditions[6])) - (Bnewweaks_plus[42] / Wk_boundary_conditions[6]));
            current_derivatives = single_var_deriv_approxes(Bnewweaks_plus, Bnew_mZ2plus, 42, BnewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus))
                            + (0.5 * Bstepplus * Bstepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > 1.0) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (Bstepplus));
            }
            tanbshift_plus = (current_derivatives[0] * Bstepplus) + (0.5 * current_derivatives[1] * Bstepplus * Bstepplus);
            
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        }
        B_TOTAL_weak_plus = Bnewweaks_plus[42] / Wk_boundary_conditions[6];

        if ((abs(B_TOTAL_weak_minus - B_weak_minus) < 1.0e-12)) {
            B_TOTAL_weak_minus = B_weak_minus + (Bstepminus / abs(mZ2shift_minus));
        }
        if ((abs(B_TOTAL_weak_plus - B_weak_plus) < 1.0e-12)) {
            B_TOTAL_weak_plus = B_weak_plus + (Bstepplus / abs(mZ2shift_plus));
        }

        std::cout << "General window established for B variation." << endl;

        return {B_weak_minus, B_weak_plus, B_TOTAL_weak_minus, B_TOTAL_weak_plus};
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error in DSN B window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]))};
    } catch (const std::exception& e) {
        std::cerr << "Numerical error in DSN B window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]))};
    } catch (...) {
        std::cerr << "Unknown error in DSN B window calculation, approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[42] / Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[42] / Wk_boundary_conditions[6]))};
    }
}

vector<double> DSN_specific_windows(vector<double>& Wk_boundary_conditions, double& current_mZ2, double& current_logQSUSY, int SpecificIndex) {
    double t_target = log(500.0);
    vector<double> pinewweaks_plus = Wk_boundary_conditions;
    vector<double> pinewweaks_minus = Wk_boundary_conditions;
    double picurrentlogQSUSY = current_logQSUSY;
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

    double piplus = pinewweaks_plus[SpecificIndex];
    double newpiplus = piplus;
    double tanbplus = pinewweaks_plus[43];
    double newtanbplus = tanbplus;

    double piminus = pinewweaks_minus[SpecificIndex];
    double newpiminus = piminus;
    double tanbminus = pinewweaks_minus[43];
    double newtanbminus = tanbminus;
    try {
        // First compute width of ABDS window
        vector<double> current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
        for (double deriv_value : current_derivatives) {
            if (isnan(deriv_value) || isinf(deriv_value)) {
                piminusEWSB = false;
            }
        }
        double pistepplus, pistepminus, bigpistep;
        if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
            pistepminus = (boost::math::float_prior(copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]);
        } else {
            pistepminus = (boost::math::float_prior((pinewweaks_minus[SpecificIndex])) - (pinewweaks_minus[SpecificIndex]));
        }    
        double mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus))
                            + (0.5 * pistepminus * pistepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus));
        }
        double tanbshift_minus = (current_derivatives[0] * pistepminus) + (0.5 * current_derivatives[1] * pistepminus * pistepminus);
        bool too_sensitive_flag_minus = false, too_sensitive_flag_plus = false;
        double pi_weak_minus, pi_weak_plus;
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
            pi_weak_minus = (pinewweaks_minus[SpecificIndex]) + pistepminus;
        } 
        while ((!too_sensitive_flag_minus) && (piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > (45.5938 * 45.5938)) && (pinew_mZ2minus < (364.7504 * 364.7504)))) {
            bigpistep = abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * pistepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = pinewweaks_minus;
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
            vector<double> pioldweaks_minus = pinewweaks_minus;
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pinewweaks_minus[SpecificIndex] = signed_square(pinewweaks_minus[SpecificIndex], bigpistep);
            } else {
                pinewweaks_minus[SpecificIndex] += bigpistep;
            }
            pinew_mZ2minus += abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * copysign(1.0, mZ2shift_minus);
            pinewweaks_minus[43] += abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((pinewweaks_minus[43] < 3.0) || (pinewweaks_minus[43] > 60.0)) {
                piminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_minus[YukIndx] *= sin(atan(pioldweaks_minus[43])) / sin(atan(pinewweaks_minus[43]));
                    } else {
                        pinewweaks_minus[YukIndx] *= cos(atan(pioldweaks_minus[43])) / cos(atan(pinewweaks_minus[43]));
                    }
                }
            }        

            if (!piminusEWSB) {
                pinewweaks_minus[SpecificIndex] = pioldweaks_minus[SpecificIndex];
                break;
            }
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pistepminus = (boost::math::float_prior(copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]);
            } else {
                pistepminus = (boost::math::float_prior((pinewweaks_minus[SpecificIndex])) - (pinewweaks_minus[SpecificIndex]));
            }      
            current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus))
                            + (0.5 * pistepminus * pistepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > abs((2.0 * sqrt(pinew_mZ2minus)) + 1.0)) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus));
            }
            tanbshift_minus = (current_derivatives[0] * pistepminus) + (0.5 * current_derivatives[1] * pistepminus * pistepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        pi_weak_minus = pinewweaks_minus[SpecificIndex];

        if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
            pistepplus = (boost::math::float_next(copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]);
        } else {
            pistepplus = (boost::math::float_next((pinewweaks_minus[SpecificIndex])) - (pinewweaks_minus[SpecificIndex]));
        }   
        current_derivatives = single_var_deriv_approxes(pinewweaks_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
        
        double mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus))
                            + (0.5 * pistepplus * pistepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        double tanbshift_plus = (current_derivatives[0] * pistepplus) + (0.5 * current_derivatives[1] * pistepplus * pistepplus);

        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus));
        }
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
            pi_weak_plus = (pinewweaks_plus[SpecificIndex]) + pistepplus;
        }
        while ((!too_sensitive_flag_plus) && (piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > (45.5938 * 45.5938)) && (pinew_mZ2plus < (364.7504 * 364.7504)))) {
            bigpistep = abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * pistepplus / abs(mZ2shift_plus);
            vector<double> checkweaksols = pinewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
            piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (piplusEWSB == true) {
                piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
            }
            piplusNoCCB = CCB_Check(checkweaksols);
            if (!(piplusEWSB) || !(piplusNoCCB)) {
                break;
            } 
            vector<double> pioldweaks_plus = pinewweaks_plus;
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pinewweaks_plus[SpecificIndex] = signed_square(pinewweaks_plus[SpecificIndex], bigpistep);
            } else {
                pinewweaks_plus[SpecificIndex] += bigpistep;
            }
            pinew_mZ2plus += abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * copysign(1.0, (-1.0) * (pinew_mZ2minus - (91.1876 * 91.1876)));
            pinewweaks_plus[43] += abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * tanbshift_plus / abs(mZ2shift_plus);
            // Now adjust Yukawas for next iteration.
            if ((pinewweaks_plus[43] < 3.0) || (pinewweaks_plus[43] > 60.0)) {
                piplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_plus[YukIndx] *= sin(atan(pioldweaks_plus[43])) / sin(atan(pinewweaks_plus[43]));
                    } else {
                        pinewweaks_plus[YukIndx] *= cos(atan(pioldweaks_plus[43])) / cos(atan(pinewweaks_plus[43]));
                    }
                }
            }        

            if (!piplusEWSB) {
                pinewweaks_plus[SpecificIndex] = pioldweaks_plus[SpecificIndex];
                break;
            }
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pistepplus = (boost::math::float_next(copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]);
            } else {
                pistepplus = (boost::math::float_next((pinewweaks_plus[SpecificIndex])) - (pinewweaks_plus[SpecificIndex]));
            }      
            current_derivatives = single_var_deriv_approxes(pinewweaks_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus))
                            + (0.5 * pistepplus * pistepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > abs((2.0 * sqrt(pinew_mZ2plus)) + 1.0)) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus));
            }
            tanbshift_plus = (current_derivatives[0] * pistepplus) + (0.5 * current_derivatives[1] * pistepplus * pistepplus);
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        }
        pi_weak_plus = pinewweaks_plus[SpecificIndex];
        
        std::cout << "ABDS window established for " << paramName << " variation." << endl;

        double pi_TOTAL_weak_minus, pi_TOTAL_weak_plus;

        if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
            pistepminus = (boost::math::float_prior(copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]);
        } else {
            pistepminus = (boost::math::float_prior((pinewweaks_minus[SpecificIndex])) - (pinewweaks_minus[SpecificIndex]));
        }     
        current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);

        mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus))
                            + (0.5 * pistepminus * pistepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_minus) > abs((2.0 * sqrt(pinew_mZ2minus)) + 1.0)) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus));
        }
        tanbshift_minus = (current_derivatives[0] * pistepminus) + (0.5 * current_derivatives[1] * pistepminus * pistepminus);
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
        } 

        while ((!too_sensitive_flag_minus) && (piminusEWSB) && (piminusNoCCB) && ((pinew_mZ2minus > 1.0)) && (pinew_mZ2minus < 1.0e12)) {
            bigpistep = abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * pistepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = pinewweaks_minus;
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
            vector<double> pioldweaks_minus = pinewweaks_minus;
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pinewweaks_minus[SpecificIndex] = signed_square(pinewweaks_minus[SpecificIndex], bigpistep);
            } else {
                pinewweaks_minus[SpecificIndex] += bigpistep;
            }
            pinew_mZ2minus += abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * copysign(1.0, mZ2shift_minus);
            pinewweaks_minus[43] += abs(((0.2 * sqrt(abs(pinew_mZ2minus))) + 0.01)) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((pinewweaks_minus[43] < 3.0) || (pinewweaks_minus[43] > 60.0)) {
                piminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_minus[YukIndx] *= sin(atan(pioldweaks_minus[43])) / sin(atan(pinewweaks_minus[43]));
                    } else {
                        pinewweaks_minus[YukIndx] *= cos(atan(pioldweaks_minus[43])) / cos(atan(pinewweaks_minus[43]));
                    }
                }
            }        

            if (!piminusEWSB) {
                pinewweaks_minus[SpecificIndex] = pioldweaks_minus[SpecificIndex];
                break;
            }
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pistepminus = (boost::math::float_prior(copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_minus[SpecificIndex])), pinewweaks_minus[SpecificIndex]);
            } else {
                pistepminus = (boost::math::float_prior((pinewweaks_minus[SpecificIndex])) - (pinewweaks_minus[SpecificIndex]));
            }      
            current_derivatives = single_var_deriv_approxes(pinewweaks_minus, pinew_mZ2minus, SpecificIndex, pinewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus))
                            + (0.5 * pistepminus * pistepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > abs((2.0 * sqrt(pinew_mZ2minus)) + 1.0)) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepminus));
            }
            tanbshift_minus = (current_derivatives[0] * pistepminus) + (0.5 * current_derivatives[1] * pistepminus * pistepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        pi_TOTAL_weak_minus = pinewweaks_minus[SpecificIndex];

        if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
            pistepplus = (boost::math::float_next(copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]);
        } else {
            pistepplus = (boost::math::float_next((pinewweaks_plus[SpecificIndex])) - (pinewweaks_plus[SpecificIndex]));
        }     
        current_derivatives = single_var_deriv_approxes(pinewweaks_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);

        mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus))
                            + (0.5 * pistepplus * pistepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus));
        }
        tanbshift_plus = (current_derivatives[0] * pistepplus) + (0.5 * current_derivatives[1] * pistepplus * pistepplus);
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
        } 

        while ((!too_sensitive_flag_plus) && (piplusEWSB) && (piplusNoCCB) && ((pinew_mZ2plus > 1.0)) && (pinew_mZ2plus < 1.0e12)) {
            bigpistep = abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * pistepplus / abs(mZ2shift_plus);
            vector<double> checkweaksols = pinewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(pinewlogQSUSY), pinew_mZ2plus);
            piplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (piplusEWSB == true) {
                piplusEWSB = Hessian_check(checkweaksols, exp(pinewlogQSUSY));
            }
            piplusNoCCB = CCB_Check(checkweaksols);
            if (!(piplusEWSB) || !(piplusNoCCB)) {
                break;
            } 
            vector<double> pioldweaks_plus = pinewweaks_plus;
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pinewweaks_plus[SpecificIndex] = signed_square(pinewweaks_plus[SpecificIndex], bigpistep);
            } else {
                pinewweaks_plus[SpecificIndex] += bigpistep;
            }
            pinew_mZ2plus += abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * copysign(1.0, (-1.0)  * (pinew_mZ2minus - (91.1876 * 91.1876)));
            pinewweaks_plus[43] += abs(((0.2 * sqrt(abs(pinew_mZ2plus))) + 0.01)) * tanbshift_plus / abs(mZ2shift_plus);
            // Now adjust Yukawas for next iteration.
            if ((pinewweaks_plus[43] < 3.0) || (pinewweaks_plus[43] > 60.0)) {
                piplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        pinewweaks_plus[YukIndx] *= sin(atan(pioldweaks_plus[43])) / sin(atan(pinewweaks_plus[43]));
                    } else {
                        pinewweaks_plus[YukIndx] *= cos(atan(pioldweaks_plus[43])) / cos(atan(pinewweaks_plus[43]));
                    }
                }
            }        

            if (!piplusEWSB) {
                pinewweaks_plus[SpecificIndex] = pioldweaks_plus[SpecificIndex];
                break;
            }
            if ((SpecificIndex >= 25) && (SpecificIndex <= 41)) {
                pistepplus = (boost::math::float_next(copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]))) - copysign(sqrt(abs(pinewweaks_plus[SpecificIndex])), pinewweaks_plus[SpecificIndex]);
            } else {
                pistepplus = (boost::math::float_next((pinewweaks_plus[SpecificIndex])) - (pinewweaks_plus[SpecificIndex]));
            }      
            current_derivatives = single_var_deriv_approxes(pinewweaks_plus, pinew_mZ2plus, SpecificIndex, pinewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus))
                            + (0.5 * pistepplus * pistepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > abs((2.0 * sqrt(pinew_mZ2plus)) + 1.0)) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (pistepplus));
            }
            tanbshift_plus = (current_derivatives[0] * pistepplus) + (0.5 * current_derivatives[1] * pistepplus * pistepplus);
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        }
        pi_TOTAL_weak_plus = pinewweaks_plus[SpecificIndex];

        if ((abs(pi_TOTAL_weak_minus - pi_weak_minus) < 1.0e-12)) {
            pi_TOTAL_weak_minus = pi_weak_minus + (pistepminus / abs(mZ2shift_minus));
        }
        if ((abs(pi_TOTAL_weak_plus - pi_weak_plus) < 1.0e-12)) {
            pi_TOTAL_weak_plus = pi_weak_plus + (pistepplus / abs(mZ2shift_plus));
        }    

        std::cout << "General window established for " << paramName << " variation." << endl;

        return {pi_weak_minus, pi_weak_plus, pi_TOTAL_weak_minus, pi_TOTAL_weak_plus};
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error in DSN " << paramName << " window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_next(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[SpecificIndex])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[SpecificIndex]))};
    } catch (const std::exception& e) {
        std::cerr << "Numerical error in DSN " << paramName << " window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_next(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[SpecificIndex])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[SpecificIndex]))};
    } catch (...) {
        std::cerr << "Unknown error in DSN " << paramName << " window calculation, approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_next(Wk_boundary_conditions[SpecificIndex]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[SpecificIndex])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[SpecificIndex]))};
    }
}

vector<double> DSN_mu_windows(vector<double>& Wk_boundary_conditions, double& current_mZ2, double& current_logQSUSY) {
    double t_target = log(500.0);
    vector<double> munewweaks_plus = Wk_boundary_conditions;
    vector<double> munewweaks_minus = Wk_boundary_conditions;
    double mucurrentlogQSUSY = current_logQSUSY;
    double munewlogQSUSY = current_logQSUSY;
    double munew_mZ2plus = current_mZ2;
    double munew_mZ2minus = current_mZ2;
    bool muminusNoCCB = true;
    bool muminusEWSB = true;
    bool muplusNoCCB = true;
    bool muplusEWSB = true;

    double muplus = munewweaks_plus[6];
    double newmuplus = muplus;
    double tanbplus = munewweaks_plus[43];
    double newtanbplus = tanbplus;

    double muminus = munewweaks_minus[6];
    double newmuminus = muminus;
    double tanbminus = munewweaks_minus[43];
    double newtanbminus = tanbminus;

    // First compute width of ABDS window
    double lambdaMu = 0.5;
    double Mu_least_Sq_Tol = 1.0e-2;
    double prev_fmu = std::numeric_limits<double>::max();
    double curr_lsq_eval = std::numeric_limits<double>::max();
    try {
        vector<double> current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);
        for (double deriv_value : current_derivatives) {
            if (isnan(deriv_value) || isinf(deriv_value)) {
                muminusEWSB = false;
            }
        }
        double mustepplus, mustepminus, bigmustep;
        mustepminus = (boost::math::float_prior(munewweaks_minus[6]) - munewweaks_minus[6]);    
        double mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus))
                            + (0.5 * mustepminus * mustepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        double original_mZ2shift_minus = mZ2shift_minus;
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus));
        }
        double tanbshift_minus = (current_derivatives[0] * mustepminus) + (0.5 * current_derivatives[1] * mustepminus * mustepminus);
        bool too_sensitive_flag_minus = false, too_sensitive_flag_plus = false;
        double mu_weak_minus, mu_weak_plus;
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
            mu_weak_minus = munewweaks_minus[6] + mustepminus;
        } 
        while ((!too_sensitive_flag_minus) && (muminusEWSB) && (muminusNoCCB) && (abs(munewweaks_minus[6]) > 1.0) && ((munew_mZ2minus > (45.5938 * 45.5938)) && (munew_mZ2minus < (364.7504 * 364.7504)))) {
            bigmustep = ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * mustepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = munewweaks_minus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
            muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (muminusEWSB == true) {
                muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
            }
            muminusNoCCB = CCB_Check(checkweaksols);
            if (!(muminusEWSB) || !(muminusNoCCB)) {
                break;
            } 
            if (!(muminusNoCCB)) {
                break;
            } 
            if (!(muminusNoCCB)) {
                break;
            } 
            vector<double> muoldweaks_minus = munewweaks_minus;
            munewweaks_minus[6] += bigmustep;
            munewweaks_minus[42] = (munewweaks_minus[42] / muoldweaks_minus[6]) * munewweaks_minus[6];
            
            if (!(muminusEWSB)) {
                munewweaks_minus[6] = muoldweaks_minus[6];
                break;
            }
            munew_mZ2minus += ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * copysign(1.0, mZ2shift_minus);
            munewweaks_minus[43] += ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((munewweaks_minus[43] < 3.0) || (munewweaks_minus[43] > 60.0)) {
                muminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_minus[YukIndx] *= sin(atan(muoldweaks_minus[43])) / sin(atan(munewweaks_minus[43]));
                    } else {
                        munewweaks_minus[YukIndx] *= cos(atan(muoldweaks_minus[43])) / cos(atan(munewweaks_minus[43]));
                    }
                }
            }        

            if (!muminusEWSB) {
                munewweaks_minus[6] = muoldweaks_minus[6];
                munewweaks_minus[42] = muoldweaks_minus[42];
                munewweaks_minus[43] = muoldweaks_minus[43];
                break;
            }
            mustepminus = (boost::math::float_prior(munewweaks_minus[6]) - munewweaks_minus[6]); 
            current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus))
                            + (0.5 * mustepminus * mustepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > 1.0) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus));
            }
            tanbshift_minus = (current_derivatives[0] * mustepminus) + (0.5 * current_derivatives[1] * mustepminus * mustepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        
        mu_weak_minus = munewweaks_minus[6];
        
        mustepplus = (boost::math::float_next(munewweaks_plus[6]) - munewweaks_plus[6]);
        current_derivatives = single_var_deriv_approxes(munewweaks_plus, munew_mZ2plus, 6, munewlogQSUSY);
        
        double mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus))
                            + (0.5 * mustepplus * mustepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        double tanbshift_plus = (current_derivatives[0] * mustepplus) + (0.5 * current_derivatives[1] * mustepplus * mustepplus);

        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus));
        }
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
            mu_weak_plus = munewweaks_plus[6] + (mustepplus / mZ2shift_plus);
        } 
        // Mu convergence becomes bad when mu is small (i.e. < 10 GeV), so cutoff at abs(mu) = 10 GeV
        while ((!too_sensitive_flag_plus) && (muplusEWSB) && (muplusNoCCB) && (abs(munewweaks_plus[6]) > 1.0) && ((munew_mZ2plus > (45.5938 * 45.5938)) && (munew_mZ2plus < (364.7504 * 364.7504)))) {
            bigmustep = ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * mustepplus / abs(mZ2shift_plus);
            vector<double> checkweaksols = munewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
            muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (muplusEWSB == true) {
                muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
            }
            muplusNoCCB = CCB_Check(checkweaksols);
            if (!(muplusEWSB) || !(muplusNoCCB)) {
                break;
            } 
            if (!(muplusNoCCB)) {
                break;
            } 
            if (!(muplusNoCCB)) {
                break;
            } 
            vector<double> muoldweaks_plus = munewweaks_plus;
            munewweaks_plus[6] += bigmustep;
            munewweaks_plus[42] = (munewweaks_plus[42] / muoldweaks_plus[6]) * munewweaks_plus[6];
            
            
            if (!(muplusEWSB)) {
                munewweaks_plus[6] = muoldweaks_plus[6];
                break;
            }
            munew_mZ2plus += ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * copysign(1.0, (-1.0) * (munew_mZ2minus - (91.1876 * 91.1876)));
            munewweaks_plus[43] += ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * tanbshift_plus / abs(mZ2shift_plus);
            // Now adjust Yukawas for next iteration.
            if ((munewweaks_plus[43] < 3.0) || (munewweaks_plus[43] > 60.0)) {
                muplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_plus[YukIndx] *= sin(atan(muoldweaks_plus[43])) / sin(atan(munewweaks_plus[43]));
                    } else {
                        munewweaks_plus[YukIndx] *= cos(atan(muoldweaks_plus[43])) / cos(atan(munewweaks_plus[43]));
                    }
                }
            }        

            if (!muplusEWSB) {
                munewweaks_plus[6] = muoldweaks_plus[6];
                munewweaks_plus[42] = muoldweaks_plus[42];
                munewweaks_plus[43] = muoldweaks_plus[43];
                break;
            }
            mustepplus = (boost::math::float_next(munewweaks_plus[6]) - munewweaks_plus[6]);    
            current_derivatives = single_var_deriv_approxes(munewweaks_plus, munew_mZ2plus, 6, munewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus))
                            + (0.5 * mustepplus * mustepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > 1.0) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus));
            }
            tanbshift_plus = (current_derivatives[0] * mustepplus) + (0.5 * current_derivatives[1] * mustepplus * mustepplus);
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        } 
        
        mu_weak_plus = munewweaks_plus[6];
        

        std::cout << "ABDS window established for mu variation." << endl;

        bool ABDSminuscheck = (muminusEWSB && muminusNoCCB); 
        bool ABDSpluscheck = (muplusEWSB && muplusNoCCB);
        bool total_ABDScheck = (ABDSminuscheck && ABDSpluscheck);
        double mu_TOTAL_weak_minus, mu_TOTAL_weak_plus;

        mustepminus = (boost::math::float_prior(munewweaks_minus[6]) - munewweaks_minus[6]); 
        current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);

        mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus))
                            + (0.5 * mustepminus * mustepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        
        if (abs(mZ2shift_minus) > 1.0) {
            mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus));
        }
        tanbshift_minus = (current_derivatives[0] * mustepminus) + (0.5 * current_derivatives[1] * mustepminus * mustepminus);
        if ((abs(mZ2shift_minus) > 1.0)) {
            too_sensitive_flag_minus = true;
        } 

        while ((!too_sensitive_flag_minus) && (muminusEWSB) && (muminusNoCCB) && (abs(munewweaks_minus[6]) > 1.0) && ((munew_mZ2minus > 1.0)) && (munew_mZ2minus < 1.0e12)) {
            bigmustep = ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * mustepminus / abs(mZ2shift_minus);
            vector<double> checkweaksols = munewweaks_minus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2minus);
            muminusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (muminusEWSB == true) {
                muminusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
            }
            muminusNoCCB = CCB_Check(checkweaksols);
            if (!(muminusEWSB) || !(muminusNoCCB)) {
                break;
            } 
            if (!(muminusNoCCB)) {
                break;
            } 
            if (!(muminusNoCCB)) {
                break;
            } 
            vector<double> muoldweaks_minus = munewweaks_minus;
            munewweaks_minus[6] += bigmustep;
            munewweaks_minus[42] = (munewweaks_minus[42] / muoldweaks_minus[6]) * munewweaks_minus[6];
            
            
            if (!(muminusEWSB)) {
                munewweaks_minus[6] = muoldweaks_minus[6];
                break;
            }
            munew_mZ2minus += ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * copysign(1.0, mZ2shift_minus);
            munewweaks_minus[43] += ((0.2 * sqrt(munew_mZ2minus)) + 0.01) * tanbshift_minus / abs(mZ2shift_minus);
            // Now adjust Yukawas for next iteration.
            if ((munewweaks_minus[43] < 3.0) || (munewweaks_minus[43] > 60.0)) {
                muminusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_minus[YukIndx] *= sin(atan(muoldweaks_minus[43])) / sin(atan(munewweaks_minus[43]));
                    } else {
                        munewweaks_minus[YukIndx] *= cos(atan(muoldweaks_minus[43])) / cos(atan(munewweaks_minus[43]));
                    }
                }
            }        

            if (!muminusEWSB) {
                munewweaks_minus[6] = muoldweaks_minus[6];
                munewweaks_minus[42] = muoldweaks_minus[42];
                munewweaks_minus[43] = muoldweaks_minus[43];
                break;
            }
            mustepminus = (boost::math::float_prior(munewweaks_minus[6]) - munewweaks_minus[6]); 
            current_derivatives = single_var_deriv_approxes(munewweaks_minus, munew_mZ2minus, 6, munewlogQSUSY);
        
            mZ2shift_minus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus))
                            + (0.5 * mustepminus * mustepminus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_minus) > 1.0) {
                mZ2shift_minus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepminus));
            }
            tanbshift_minus = (current_derivatives[0] * mustepminus) + (0.5 * current_derivatives[1] * mustepminus * mustepminus);
            if ((abs(mZ2shift_minus) > 1.0)) {
                too_sensitive_flag_minus = true;
            } 
        }
        mu_TOTAL_weak_minus = munewweaks_minus[6];
        mustepplus = (boost::math::float_next(munewweaks_plus[6]) - munewweaks_plus[6]);
        current_derivatives = single_var_deriv_approxes(munewweaks_plus, munew_mZ2plus, 6, munewlogQSUSY);

        mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus))
                            + (0.5 * mustepplus * mustepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
        if (abs(mZ2shift_plus) > 1.0) {
            mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus));
        }
        tanbshift_plus = (current_derivatives[0] * mustepplus) + (0.5 * current_derivatives[1] * mustepplus * mustepplus);
        if ((abs(mZ2shift_plus) > 1.0)) {
            too_sensitive_flag_plus = true;
            mu_TOTAL_weak_plus = munewweaks_plus[6] + (mustepplus / abs(mZ2shift_plus));
        } 

        while ((!too_sensitive_flag_plus) && (muplusEWSB) && (muplusNoCCB) && (abs(munewweaks_plus[6]) > 1.0) && ((munew_mZ2plus > 1.0)) && (munew_mZ2plus < 1.0e12)) {
            bigmustep = ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * mustepplus / abs(mZ2shift_plus);
            vector<double> checkweaksols = munewweaks_plus;
            vector<double> checkRadCorrs = radcorr_calc(checkweaksols, exp(munewlogQSUSY), munew_mZ2plus);
            muplusEWSB = EWSB_Check(checkweaksols, checkRadCorrs);
            // Checking loop-level EWSB
            if (muplusEWSB == true) {
                muplusEWSB = Hessian_check(checkweaksols, exp(munewlogQSUSY));
            }
            muplusNoCCB = CCB_Check(checkweaksols);
            if (!(muplusEWSB) || !(muplusNoCCB)) {
                break;
            } 
            if (!(muplusNoCCB)) {
                break;
            } 
            if (!(muplusNoCCB)) {
                break;
            } 
            vector<double> muoldweaks_plus = munewweaks_plus;
            munewweaks_plus[6] += bigmustep;
            munewweaks_plus[42] = (munewweaks_plus[42] / muoldweaks_plus[6]) * munewweaks_plus[6];
            
            
            if (!(muplusEWSB)) {
                munewweaks_plus[6] = muoldweaks_plus[6];
                break;
            }
            munew_mZ2plus += ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * copysign(1.0, (-1.0) * (munew_mZ2minus - (91.1876 * 91.1876)));
            munewweaks_plus[43] += ((0.2 * sqrt(munew_mZ2plus)) + 0.01) * tanbshift_plus / abs(mZ2shift_plus);
            // Now adjust Yukawas for next iteration.
            if ((munewweaks_plus[43] < 3.0) || (munewweaks_plus[43] > 60.0)) {
                muplusEWSB = false;
            } else {
                for (int YukIndx = 7; YukIndx < 16; ++YukIndx) {
                    if ((YukIndx >=7) && (YukIndx < 10)) {
                        munewweaks_plus[YukIndx] *= sin(atan(muoldweaks_plus[43])) / sin(atan(munewweaks_plus[43]));
                    } else {
                        munewweaks_plus[YukIndx] *= cos(atan(muoldweaks_plus[43])) / cos(atan(munewweaks_plus[43]));
                    }
                }
            }        

            if (!muplusEWSB) {
                munewweaks_plus[6] = muoldweaks_plus[6];
                munewweaks_plus[42] = muoldweaks_plus[42];
                munewweaks_plus[43] = muoldweaks_plus[43];
                break;
            }
            mustepplus = (boost::math::float_next(munewweaks_plus[6]) - munewweaks_plus[6]);    
            current_derivatives = single_var_deriv_approxes(munewweaks_plus, munew_mZ2plus, 6, munewlogQSUSY);
        
            mZ2shift_plus = ((((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus))
                            + (0.5 * mustepplus * mustepplus * ((current_derivatives[1] * current_derivatives[2])
                                                        + (current_derivatives[0] * current_derivatives[0] * current_derivatives[4])
                                                        + (2.0 * current_derivatives[0] * current_derivatives[5]) + current_derivatives[6])));
            if (abs(mZ2shift_plus) > 1.0) {
                mZ2shift_plus = (((current_derivatives[0] * current_derivatives[2]) + current_derivatives[3]) * (mustepplus));
            }
            tanbshift_plus = (current_derivatives[0] * mustepplus) + (0.5 * current_derivatives[1] * mustepplus * mustepplus);
            if ((abs(mZ2shift_plus) > 1.0)) {
                too_sensitive_flag_plus = true;
            } 
        }
        mu_TOTAL_weak_plus = munewweaks_plus[6];

        if ((abs(mu_TOTAL_weak_minus - mu_weak_minus) < 1.0e-12)) {
            mu_TOTAL_weak_minus = mu_weak_minus + (mustepminus / abs(mZ2shift_minus));
        }
        if ((abs(mu_TOTAL_weak_plus - mu_weak_plus) < 1.0e-12)) {
            mu_TOTAL_weak_plus = mu_weak_plus + (mustepplus / abs(mZ2shift_plus));
        }        

        std::cout << "General window established for mu variation." << endl;

        return {mu_weak_minus, mu_weak_plus, mu_TOTAL_weak_minus, mu_TOTAL_weak_plus};
    } catch (const std::runtime_error& e) {
        std::cerr << "Runtime error in DSN mu window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[6]))};
    } catch (const std::exception& e) {
        std::cerr << "Numerical error in DSN mu window calculation: " << e.what() << std::endl;
        std::cerr << "Approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[6]))};
    } catch (...) {
        std::cerr << "Unknown error in DSN mu window calculation, approximating solution." << std::endl;
        return {boost::math::float_prior(Wk_boundary_conditions[6]),
                boost::math::float_next(Wk_boundary_conditions[6]),
                boost::math::float_prior(boost::math::float_prior(Wk_boundary_conditions[6])),
                boost::math::float_next(boost::math::float_next(Wk_boundary_conditions[6]))};
    }
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
            newterm = abs(log10(1.0 + (numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]))))\
                / abs(numeric_limits<double>::epsilon() * abs(GUT_boundary_conditions[6]));
            DSN_higgsino = 1.0 / abs(((pow(10.0, 0.5) - pow(10.0, -0.5))) * abs(GUT_boundary_conditions[6]));
        }
        DSN += abs(log10(abs(DSN_higgsino)) - log10(abs(newterm)));
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
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[42] / GUT_boundary_conditions[6]) - boost::math::float_prior(GUT_boundary_conditions[42] / GUT_boundary_conditions[6])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[42] / GUT_boundary_conditions[6], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
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
            newterm = (soft_prob_calc(copysign(sqrt(abs(boost::math::float_next(GUT_boundary_conditions[25]))), boost::math::float_next(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(copysign(sqrt(abs(boost::math::float_prior(GUT_boundary_conditions[25]))), boost::math::float_prior(GUT_boundary_conditions[25])), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25])) - boost::math::float_prior(copysign(sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]))));
            DSN_soft_num = soft_prob_calc(copysign(pow(10.0, 0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(copysign(pow(10.0, -0.5) * sqrt(abs(GUT_boundary_conditions[25])), GUT_boundary_conditions[25]), (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
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
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[3]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[3]) - boost::math::float_prior(GUT_boundary_conditions[3])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[3], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
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
            newterm = (soft_prob_calc(boost::math::float_next(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0)
                        - soft_prob_calc(boost::math::float_prior(GUT_boundary_conditions[16]), (2.0 * nF) + (1.0 * nD) - 1.0))\
                / (abs(boost::math::float_next(GUT_boundary_conditions[16]) - boost::math::float_prior(GUT_boundary_conditions[16])));
            DSN_soft_num = soft_prob_calc(pow(10.0, 0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0)\
                - soft_prob_calc(pow(10.0, -0.5) * GUT_boundary_conditions[16], (2.0 * nF) + (1.0 * nD) - 1.0);
        }
        DSN += abs(log10(abs(DSN_soft_num / DSN_soft_denom)) - log10(abs(newterm)));
    }

    return DSN;
}

double DSN_calc(int precselno, std::vector<double> Wk_boundary_conditions,
                double& current_mZ2, double& current_logQSUSY,
                double& current_logQGUT, int& nF, int& nD) {
    // ThreadPool pool(10);
    // vector<std::future<double>> results;
    // vector<int> paramselnoList = {3, 4, 5, 6, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};
    // double runningSumDSN = 0.0;
    // mutex sumMutex;

    // for (int paramNo : paramselnoList) {
    //     auto task = pool.enqueue(
    //         [paramNo, &GUT_boundary_conditions, current_mZ2, current_logQSUSY, current_logQGUT]() {
    //             return DSN_term(paramNo, GUT_boundary_conditions, current_mZ2, current_logQSUSY, current_logQGUT);
    //         }
    //     );
    //     results.push_back(move(task));
    // }

    // for (auto& future : results) {
    //     double taskResult = future.get();
    //     lock_guard<mutex> lock(sumMutex);
    //     runningSumDSN += taskResult;
    // }
    double runningSumDSN = 0.0;
    runningSumDSN += DSN_term(6, GUT_boundary_conditions, current_mZ2, current_logQSUSY, current_logQGUT);
    return runningSumDSN;
}
