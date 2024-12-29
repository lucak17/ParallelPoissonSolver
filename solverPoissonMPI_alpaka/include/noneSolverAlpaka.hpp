#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once


#include <alpaka/alpaka.hpp>
#include "iterativeSolverBaseAlpaka.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"


template <int DIM,typename T_data, typename T_data_noneSolver, int tolerance, int maxIteration>
class NoneSolverAlpaka : public IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>{
    public:

    NoneSolverAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                CommunicatorMPI<DIM,T_data>& communicatorMPI,  const AlpakaHelper<DIM,T_data>& alpakaHelper):
       IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper)
    {
    }

    inline void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx> bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx> bufB)
    {
        memcpy(this->queueSolver_, bufX, bufB, this->alpakaHelper_.extent_);
    }

    private:
};