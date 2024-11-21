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

using T_NoneSolver = NoneSolver<DIM,T_data,tollPreconditionerSolver,iterMaxPreconditioner>;

//using T_Preconditioner = NoneSolver<DIM,T_data,tollPreconditionerSolver,iterMaxPreconditioner>;
//using T_Preconditioner = BiCGSTAB<DIM, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, !communicationON, T_NoneSolver>;
using T_Preconditioner2 = ChebyshevIteration<DIM,T_data,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolver>;
using T_Preconditioner3 = BaseCG<DIM, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, communicationOFF, T_Preconditioner2>;
//using T_Preconditioner = BiCGSTABLocal<DIM, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, communicationOFF, T_NoneSolver>;
using T_Preconditioner = BiCGSTAB<DIM, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, communicationOFF, T_NoneSolver>;
//using T_Solver = BiCGSTAB<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_NoneSolver>;
using T_Solver = BiCGSTAB<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_Preconditioner2>;
//using T_Solver = BiCGSTABLocal<DIM, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, !communicationON, T_NoneSolver>;


// X Y Z
constexpr std::array<int,3> npglobal={32,32,64};
constexpr std::array<T_data,3> ds={0.1,0.1,0.1};
constexpr std::array<T_data,3> origin={0,0,0};
constexpr std::array<int,3> guards={1,1,1};
constexpr std::array<int,6> bcsType={0,1,0,1,0,1}; //0 Dirichlet, 1 Neumann, -1 No dim; order x-x+ y-y+ z-z+
constexpr std::array<T_data,6> bcsValue={0,0,0,0,0,0}; //0 Dirichlet, 1 Neumann, -1 No dim



#endif
