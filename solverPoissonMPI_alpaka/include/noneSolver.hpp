#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once


#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "iterativeSolverBase.hpp"
#include "communicationMPI.hpp"


template <int DIM,typename T_data, int tolerance, int maxIteration>
class NoneSolver : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    NoneSolver(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, CommunicatorMPI<DIM,T_data>& communicatorMPI):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI)
    {
    }

    inline void operator()(T_data fieldX[], T_data fieldB[], MatrixFreeOperatorA<DIM,T_data>& operatorA)
    {
        std::memcpy(fieldX, fieldB, this->ntotlocal_guards_*sizeof(T_data));
    }

};