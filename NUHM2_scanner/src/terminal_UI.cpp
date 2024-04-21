#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
#include <iomanip>
#include <limits>
#include <regex>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include "mZ_numsolver.hpp"
#include "terminal_UI.hpp"
#include "MSSM_RGE_solver.hpp"
#include "MSSM_RGE_solver_with_stopfinder.hpp"
#include "DEW_calc.hpp"
#include "DBG_calc.hpp"
#include "DHS_calc.hpp"
#include "DSN_calc.hpp"
#include "radcorr_calc.hpp"
#include "slhaea.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using namespace SLHAea;
namespace fs = filesystem;

double getRenormalizationScale(const Coll& slha, const string& blockName) {
    double scale = 2000.0; // Default scale value if not found

    if (slha.find(blockName) != slha.end()) {
        for (const auto& line : slha.at(blockName)) {
            // Convert the line to a string for regex search
            string lineStr = to_string(line);
            smatch match;
            // Regex to find 'Q=' followed by a number (the scale)
            regex scaleRegex("Q= ([\\d\\.eE\\-\\+]+)");

            if (regex_search(lineStr, match, scaleRegex) && match.size() > 1) {
                // Convert the first captured group to a double
                scale = stod(match.str(1));
                break; // Assuming we only need the first occurrence
            }
        }
    }

    return scale;
}

tuple<double, double, double, double, double, double, double> doPoint(const fs::path& filePath, double& m0_Val, double& mhf_Val) {
    std::cout << fixed << setprecision(9);

    int printPrec = 9;
    
    /******************************************************************
     ********************* DBG MODEL SELECTION ************************
    ******************************************************************/
    int modinp = 1;
    int precinp = 3;
    
    /******************************************************************
     ********************** DSN Configuration *************************
    ******************************************************************/
    
    int nF_input = 1;
    int nD_input = 0;
            
    /******************************************************************
     ************************ SLHA READ-IN ****************************
    ******************************************************************/
    
    ifstream ifs(filePath);

    if (!ifs.good()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return {m0_Val, mhf_Val, NAN, NAN, NAN, NAN, NAN}; // Adjust according to how you want to handle file read errors
    }

    Coll input(ifs);

    double mZ = 91.1876;

    auto getDoubleVecValue = [&](const string& block, int i, double defaultValue = 0.0) -> double {
        try {
            return to<double>(input.at(block).at(to_string(i)).at(1));
        } catch (const exception& e) {
            return defaultValue;
        }
    };

    auto getStringValue = [&](const string& block, int key, const string& defaultValue = "") -> string {
        try {
            // Assuming that the value of interest is in the second position (index 1 when accessed as a vector),
            // concatenate all elements from index 1 to the end to form the full string.
            const auto& line = input.at(block).at(to_string(key));
            string fullString;
            for (size_t i = 1; i < line.size(); ++i) {
                if (i > 1) fullString += " "; // Add space between elements if not the first element
                fullString += line[i];
            }
            return fullString;
        } catch (const exception& e) {
            return defaultValue;
        }
    };

    string spinfoValue = getStringValue("SPINFO", 4);
    std::cout << spinfoValue << endl;
    if (spinfoValue.find("Point invalid") != string::npos) {
        // The string contains "Point invalid", so handle accordingly
        return {m0_Val, mhf_Val, NAN, NAN, NAN, NAN, NAN};
    }

    auto getDoubleMatValue = [&](const string& block, int i, int j, double defaultValue = 0.0) -> double {
        try {
            return to<double>(input.at(block).at(i, j).at(2));
        } catch (const exception& e) {
            return defaultValue;
        }
    };
    // Higgs sector variables
    double vHiggs = getDoubleVecValue("HMIX", 3);
    double tanb = getDoubleVecValue("HMIX", 2);
    double beta = atan(tanb);
    double muQ = getDoubleVecValue("HMIX", 1);
    // Yukawas (2nd and 1st gens approximated if not present)
    double y_t = getDoubleMatValue("YU",3,3);
    double y_c = getDoubleMatValue("YU",2,2);
    if (y_c == 0.0) {
        y_c = 0.003882759826930082 * y_t;
    }
    double y_u = getDoubleMatValue("YU",1,1);
    if (y_u == 0.0) {
        y_u = 7.779613278615955e-6 * y_t;
    }
    double y_b = getDoubleMatValue("YD",3,3);
    double y_s = getDoubleMatValue("YD",2,2);
    if (y_s == 0.0) {
        y_s = 0.0206648802754076 * y_b;
    }
    double y_d = getDoubleMatValue("YD",1,1);
    if (y_d == 0.0) {
        y_d = 0.0010117174290779725 * y_b;
    }
    double y_tau = getDoubleMatValue("YE",3,3);
    double y_mu = getDoubleMatValue("YE",2,2);
    if (y_mu == 0.0) {
        y_mu = 0.05792142442492775 * y_tau;
    }
    double y_e = getDoubleMatValue("YE",1,1);
    if (y_e == 0.0) {
        y_e = 0.0002801267571260388 * y_tau;
    }
    // Gauge couplings
    double g_pr = getDoubleVecValue("GAUGE", 1);
    double g_2 = getDoubleVecValue("GAUGE", 2);
    double g_s = getDoubleVecValue("GAUGE", 3);
    // Soft trilinear couplings
    // Check for which soft trilinear block is present
    // softTrilinIdentif: 0 = "TU,TD,TE", 1 = "AU, AD, AE"
    int softTrilinIdentif = 0;
    string softTrilinUBlock, softTrilinDBlock, softTrilinEBlock;
    double a_t = (-1.6) * m0_Val * y_t;
    double a_c = (-1.6) * m0_Val * y_c;
    double a_u = (-1.6) * m0_Val * y_u;
    double a_b = (-1.6) * m0_Val * y_b;
    double a_s = (-1.6) * m0_Val * y_s;
    double a_d = (-1.6) * m0_Val * y_d;
    double a_tau = (-1.6) * m0_Val * y_tau;
    double a_mu = (-1.6) * m0_Val * y_mu;
    double a_e = (-1.6) * m0_Val * y_e;
    // Gaugino masses
    double my_M1, my_M2, my_M3;
    my_M1 = mhf_Val;
    my_M2 = mhf_Val;
    my_M3 = mhf_Val;
    // Soft Higgs masses
    double mHusq, mHdsq;
    mHusq = getDoubleVecValue("MSOFT", 22);
    mHdsq = getDoubleVecValue("MSOFT", 21);
    // Soft scalar masses
    double mQ3sq = m0_Val * m0_Val;
    double mQ2sq = m0_Val * m0_Val;
    double mQ1sq = m0_Val * m0_Val;
    double mL3sq = m0_Val * m0_Val, mL2sq = m0_Val * m0_Val, mL1sq = m0_Val * m0_Val;
    double mU3sq = m0_Val * m0_Val, mU2sq = m0_Val * m0_Val, mU1sq = m0_Val * m0_Val;
    double mD3sq = m0_Val * m0_Val, mD2sq = m0_Val * m0_Val, mD1sq = m0_Val * m0_Val;
    double mE3sq = m0_Val * m0_Val, mE2sq = m0_Val * m0_Val, mE1sq = m0_Val * m0_Val;
    double mh0 = getDoubleVecValue("MASS", 25);
    double mgl = getDoubleVecValue("MASS", 1000021);
    double mC1 = getDoubleVecValue("MASS", 1000024);
    double mstop1 = getDoubleVecValue("MASS", 1000006);
    double SLHA_GUT_scale = getRenormalizationScale(input, "GAUGE");
    /* Use 2-loop MSSM RGEs to evolve results to a renormalization scale of 
        Q = sqrt(mst1 * mst2) if the submitted SLHA file is not currently at that scale.
        This is so evaluations of the naturalness measures are always performed
        at a scale that somewhat minimizes logs and to avoid badly organized SLHA files.
        ///////////////////////////////////////////////////////////////////////
        The result is then run to a high scale of 3*10^16 GeV, and an approximate GUT
        scale is chosen at the value where g1(Q) is closest to g2(Q) over the scanned
        renormalization scales. This is done by iterating and adjusting GUT thresholds to 
        account for log corrections at that scale. 
        ///////////////////////////////////////////////////////////////////////
        This running to the GUT scale is used in the evaluations of Delta_HS and Delta_BG.
        Compute loop-level soft Higgs bilinear parameter b=B*mu at SUSY scale for RGE BC
        after. 
    */        

    /******************************************************************
     ***************** ESTABLISH WEAK-SCALE VALUES ********************
        ******************************************************************/

    vector<double> mySLHABCs;
    mySLHABCs = {sqrt(5.0 / 3.0) * g_pr, g_2, g_s, my_M1, my_M2, my_M3,
                    muQ, y_t, y_c, y_u, y_b, y_s, y_d, y_tau, y_mu, y_e,
                    a_t, a_c, a_u, a_b, a_s, a_d, a_tau, a_mu, a_e,
                    mHusq, mHdsq, mQ1sq, mQ2sq, mQ3sq, mL1sq, mL2sq,
                    mL3sq, mU1sq, mU2sq, mU3sq, mD1sq, mD2sq, mD3sq,
                    mE1sq, mE2sq, mE3sq, 0.0, tanb};
    for (double value : mySLHABCs) {
        std::cout << value << endl;
    }
    // SUSY scale equal to Q = sqrt(mt1(Q) * mt2(Q))
    double tempT_target = log(200.0); 
    vector<RGEStruct> SUSYscale_struct = solveODEstoMSUSY(mySLHABCs, log(SLHA_GUT_scale), -1.0e-6, tempT_target, 91.1876 * 91.1876);

    double SLHAQSUSY = exp(SUSYscale_struct[0].SUSYscale_eval);
    vector<double> first_SUSY_BCs = SUSYscale_struct[0].RGEsolvec;
    vector<double> first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
    tanb = first_SUSY_BCs[43];
    mHdsq = first_SUSY_BCs[26];
    mHusq = first_SUSY_BCs[25];
    muQ = first_SUSY_BCs[6];
    // Converge a value of mu that gives mZ=91.1876 GeV
    double lsqtol = 1.0e-8;
    double curr_iter_lsq = 100.0;
    double muQsq = muQ * muQ;
    double newmuQsq = muQsq;
    while (curr_iter_lsq > lsqtol) {
        newmuQsq = ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)) - (91.1876 * 91.1876 / 2.0);
        first_SUSY_BCs[6] = copysign(sqrt(abs(newmuQsq)), muQ);
        first_radcorrs = radcorr_calc(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);
        curr_iter_lsq = pow((muQsq) - (newmuQsq), 2.0);
        muQsq = newmuQsq;
    }
    double currentmZ2 = ((2.0 * ((mHdsq + first_radcorrs[1] - ((mHusq + first_radcorrs[0]) * pow(tanb, 2.0))) / (pow(tanb, 2.0) - 1.0)))
                            - (2.0 * muQsq));
    double getmZ2_value = getmZ2(first_SUSY_BCs, SLHAQSUSY, 91.1876 * 91.1876);

    // Now we calculate the value of b=B*mu coming from this SLHA point. 
    double BmuSLHA = sin(2.0 * beta) * (mHusq + first_radcorrs[0] + mHdsq + first_radcorrs[1] + (2.0 * muQsq)) / 2.0;
    //std::cout << BmuSLHA << endl;
    first_SUSY_BCs[42] = BmuSLHA;

    /******************************************************************
     ******************** ESTABLISH GUT VALUES ************************
        ******************************************************************/
    
    // Get GUT scale now
    vector<double> first_GUT_BCs = solveODEs(first_SUSY_BCs, log(SLHAQSUSY), log(SLHA_GUT_scale), 1.0e-6);
    double curr_iter_QGUT = log(SLHA_GUT_scale);

    // /******************************************************************
    //  ********************* COMPUTE DEW VALUES *************************
    //     ******************************************************************/

    // vector<LabeledValue> dewlist = DEW_calc(first_SUSY_BCs, SLHAQSUSY);
    // double DEW = dewlist[0].value;
    
    // /******************************************************************
    //  ********************* COMPUTE DHS VALUES *************************
    //     ******************************************************************/

    // vector<LabeledValueHS> dhslist = DHS_calc(first_GUT_BCs[26], first_SUSY_BCs[26] - first_GUT_BCs[26],
    //                                             first_GUT_BCs[25], first_SUSY_BCs[25] - first_GUT_BCs[25],
    //                                             pow(first_GUT_BCs[6], 2.0),
    //                                             pow(first_SUSY_BCs[6], 2.0) - pow(first_GUT_BCs[6], 2.0),
    //                                             91.1876 * 91.1876, first_SUSY_BCs[43] * first_SUSY_BCs[43], first_radcorrs[0], first_radcorrs[1]);

    // double DHS = dhslist[0].value;

    // /******************************************************************
    //  ********************* COMPUTE DBG VALUES *************************
    //     ******************************************************************/

    double logQSUSY = log(SLHAQSUSY);
    // vector<LabeledValueBG> myDBGlist = DBG_calc(modinp, precinp, curr_iter_QGUT,
    //                                             logQSUSY, tanb, first_GUT_BCs, currentmZ2);
    // double DBG = myDBGlist[0].value;
    
    /******************************************************************
     ********************* COMPUTE DSN VALUES *************************
        ******************************************************************/

    double DSN = DSN_calc(first_GUT_BCs, getmZ2_value, logQSUSY, curr_iter_QGUT);
    std::cout << "-------------------------------------------------------------------------" << endl;
    return {m0_Val, mhf_Val, mgl, mh0, mC1, mstop1, DSN};
}

int main() {
    vector<double> m0Values(32), mhfValues(32);
    double m0Start = 100.0, m0End = 9000.0, mhfStart = 100.0, mhfEnd = 2000.0;
    double m0Step = (m0End - m0Start) / 31.0;
    double mhfStep = (mhfEnd - mhfStart) / 31.0;

    for (int ii = 0; ii < 32; ++ii) {
        m0Values[ii] = m0Start + (ii * m0Step);
        mhfValues[ii] = mhfStart + (ii * mhfStep);
    }

    fs::path dirPath("/home/dakotah/softsusy/softsusy-4.1.13/NUHM2_tests/");

    ofstream csvFile("/mnt/c/Users/dakot/Documents/Research/DSN4SLHA/NUHM2_m0_vs_mhf_scan/NUHM2scanResults.csv");

    csvFile << "FileNo,m0,mhf,mgl,mh0,mC1,mstop1,DSN\n";
    csvFile.flush();

    regex filePattern("spectrum_(\\d+).out");
    smatch match;

    int FilesDoneCounter = 0;

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        std::string fileName = entry.path().filename().string();

        if (std::regex_match(fileName, match, filePattern)) {
            int fileNumber = std::stoi(match[1].str());
            // Calculate the indices for m0Values and mhfValues based on fileNumber
            // Assuming fileNumber ranges from 1 to 1024 and maps directly to combinations of m0 and mhf values
            int m0Index = (fileNumber - 1) / 32; // Integer division to determine row
            int mhfIndex = (fileNumber - 1) % 32; // Remainder to determine column

            double m0Value = m0Values[m0Index];
            double mhfValue = mhfValues[mhfIndex];

            auto [current_M0, current_MHF, mgl, mh0, mC1, mstop1, DSN] = doPoint(entry.path(), m0Value, mhfValue);
            csvFile << fileNumber << "," << m0Value << "," << mhfValue << "," << mgl << "," << mh0 << "," << mC1 << "," << mstop1 << "," << DSN << "\n";
            csvFile.flush();
            FilesDoneCounter++;
            std::cout << FilesDoneCounter << " files done." << endl;
        }
    }

    csvFile.close();
    return 0;
}