#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once


#include <alpaka/alpaka.hpp>
#include "iterativeSolverBase.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpaka.hpp"


template <int DIM,typename T_data, int tolerance, int maxIteration>
class NoneSolverAlpaka : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    NoneSolverAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
       IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI),
       alpakaHelper_(alpakaHelper)
    {
    }

    inline void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx> bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx> bufB)
    {
        alpaka::Queue<Acc, alpaka::Blocking> queue{this->alpakaHelper_.devAcc_};
        memcpy(queue, bufX, bufB, this->alpakaHelper_.extent_);
    }

    private:
    const AlpakaHelper<DIM,T_data>& alpakaHelper_;

};