#ifndef PARAM_HPP
#define PARAM_HPP

#include <iostream>
#include <chrono>
#include <iomanip> 
#include <ctime>
#include <array>
#include <cstring>
#include <cmath>

#pragma once
#include "solvers.hpp"




constexpr bool ischebyshevMainLoop = false;
constexpr bool isbiCGMainLoop1 = true;
constexpr bool isbiCGMainLoop2 = false;
constexpr bool communicationON = true;
constexpr bool communicationOFF = false;
//constexpr bool alpakaAcc = true;

using T_NoneSolverAlpaka = NoneSolverAlpaka<DIM,T_data,tollPreconditionerSolver,iterMaxPreconditioner>;

using T_PreconditionerAlpaka = ChebyshevIterationAlpaka<DIM,T_data,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolverAlpaka>;

using T_Solver = BiCGstabAlpaka<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_PreconditionerAlpaka>;

// X Y Z
constexpr std::array<int,3> npglobal={128,128,128};
constexpr std::array<T_data,3> ds={0.1,0.1,0.1};
constexpr std::array<T_data,3> origin={3,2.5,10};
constexpr std::array<int,6> bcsType={0,1,1,0,1,0}; //0 Dirichlet, 1 Neumann, -1 No dim; order x-x+ y-y+ z-z+
constexpr std::array<T_data,6> bcsValue={0,0,0,0,0,0}; //0 Dirichlet, 1 Neumann, -1 No dim



#endif
