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


constexpr bool ischebyshevMainLoop = false; // use to chose if solver is main solver (true) or preconditioner (false)
constexpr bool isbiCGMainLoop1 = true; // use to chose if solver is main solver (true) or preconditioner (false)
constexpr bool isbiCGMainLoop2 = false; // use to chose if solver is main solver (true) or preconditioner (false)
constexpr bool communicationON = true; // allows communications between subdomains in the solver
constexpr bool communicationOFF = false; // no communications between subdomains in the solver
constexpr bool global = true; // use to chose if chebyshev iteration uses local ( and not rescaled) or global (and rescaled) eigenvalues - if chebyshev iteration is used as main solver this must be set to true and the rescaling factors for the eigenvalues must be manually set to 1
constexpr bool local = false; // use to chose if chebyshev iteration uses local ( and not rescaled) or global (and rescaled) eigenvalues
//constexpr bool alpakaAcc = true;

using T_NoneSolverAlpaka = NoneSolverAlpaka<DIM,T_data,T_data,tollPreconditionerSolver,iterMaxPreconditioner>; // no preconditioner --> use this as preconditioner to run with basic BiCGS

using T_PreconditionerChebLocal = ChebyshevIterationAlpaka<DIM,T_data,T_data_chebyshev,local,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolverAlpaka>;

using T_PreconditionerChebGlobal = ChebyshevIterationAlpaka<DIM,T_data,T_data_chebyshev,global,tollPreconditionerSolver,chebyshevMax, ischebyshevMainLoop, communicationOFF, T_NoneSolverAlpaka>;

using T_PreconditionerBiCGStabLocal = BiCGstabAlpaka<DIM, T_data, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, communicationOFF, T_NoneSolverAlpaka>;

using T_PreconditionerBiCGStabGlobal = BiCGstabAlpaka<DIM, T_data, T_data, tollPreconditionerSolver, iterMaxPreconditioner, isbiCGMainLoop2, communicationON, T_NoneSolverAlpaka>;

// main solver - the preconditioner is the last template parameter
using T_Solver = BiCGstabAlpaka<DIM, T_data, T_data, tollMainSolver, iterMaxMainSolver, isbiCGMainLoop1, communicationON, T_PreconditionerChebGlobal>;
//using T_Solver = ChebyshevIterationAlpaka<DIM,T_data,T_data_chebyshev,global,tollPreconditionerSolver,chebyshevMax, true, communicationON, T_NoneSolverAlpaka>;

// residualHistory.txt has the following structur:
/*
Time to solution in seconds
Total number of main solver iterations
Total time or preconditioner iterations (already taken into account that the preconditioner is applied twice per main cycle)
L2 norm of the residual vector at each main solver iteration
*/
constexpr bool writeResidual = false; // if true write residual history in residualHistory.txt file (must set trackErrorFromIterationHistory=1 in solverSetup.hpp)
constexpr bool writeSolution = false; // if true write solution to solution.dat in binary



// Domain dimension and grid definition
// order of the dimensions: X Y Z
constexpr std::array<int,3> npglobal={256,256,256}; // total number of grid points in x,y,z
constexpr std::array<T_data,3> ds={0.1,0.1,0.1}; // mesh resolution dx, dy, dz
constexpr std::array<T_data,3> origin={3,2.5,10}; // origin of the cartesian grid - the grid extends for origin to origin+ds*(npglobal - 1)
constexpr std::array<int,6> bcsType={0,1,1,0,1,0}; // Boundary conditions -> 0 Dirichlet, 1 Neumann, -1 No dim; (use only 0 or 1) - order x-x+ y-y+ z-z+
constexpr std::array<T_data,6> bcsValue={0,0,0,0,0,0}; //0 Dirichlet, 1 Neumann, -1 No dim --> keep always to 0, BCs values are taken from ExactSolutionAndBCs in solverSetap.hpp

#endif
