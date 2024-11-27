#ifndef PARAM_HPP
#define PARAM_HPP

#include <iostream>
#include <chrono>
#include <iomanip>  // For std::put_time
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

using T_NoneSolver = NoneSolver<DIM,T_data,tollPreconditionerSolver,iterMaxPreconditioner>;
using T_NoneSolverAlpaka = NoneSolverAlpaka<DIM,T_data,tollPreconditionerSolver,iterMaxPreconditioner>;

using T_PreconditionerAlpaka = ChebyshevIterationAlpaka<DIM,T_data,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolver>;
using T_Preconditioner = ChebyshevIteration<DIM,T_data,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolver>;
//using T_Solver = BiCGSTAB<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_NoneSolver>;
//using T_Solver = BiCGSTABLocal<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, !communicationON, T_NoneSolver>;
using T_Solver = BiCGstabAlpaka<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_PreconditionerAlpaka>;
//using T_Solver = ChebyshevIterationAlpaka2<DIM, T_data, tollMainSolver, chebyshevMax, isbiCGMainLoop1, communicationOFF, T_NoneSolver>;
//using T_Solver = ChebyshevIteration<DIM,T_data,tollPreconditionerSolver,chebyshevMax, isbiCGMainLoop1, communicationOFF, T_NoneSolver>;
// X Y Z
constexpr std::array<int,3> npglobal={512,512,512};
constexpr std::array<T_data,3> ds={0.1,0.1,0.1};
constexpr std::array<T_data,3> origin={0,0,0};
constexpr std::array<int,3> guards={1,1,1};
constexpr std::array<int,6> bcsType={0,0,0,0,0,0}; //0 Dirichlet, 1 Neumann, -1 No dim; order x-x+ y-y+ z-z+
constexpr std::array<T_data,6> bcsValue={0,0,0,0,0,0}; //0 Dirichlet, 1 Neumann, -1 No dim



#endif
