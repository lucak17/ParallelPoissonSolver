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


template <int DIM, typename T_data, int maxIteration>
class TestAcc : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    TestAcc(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, CommunicatorMPI<DIM,T_data>& communicatorMPI):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI)
    {

    }

    void operator()(const BlockGrid<DIM,T_data>& blockGrid, T_data fieldX[], T_data fieldB[])
    {
        // alpaka offloading
        // Select specific devices
        auto const platformHost = alpaka::PlatformCpu{};
        auto const devHost = alpaka::getDevByIdx(platformHost, 0);
        auto const platformAcc = alpaka::Platform<Acc>{};
        // get suitable device for this Acc
        auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);

        if constexpr (DIM==3){
            // order Y X Z
            const alpaka::Vec<alpaka::DimInt<3u>, Idx> numNodes{blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0], blockGrid.getNlocalNoGuards()[2]};
            const alpaka::Vec<alpaka::DimInt<3u>, Idx> haloSize{blockGrid.getDs()[1], blockGrid.getDs()[0],blockGrid.getDs()[2]};
            const alpaka::Vec<alpaka::DimInt<3u>, Idx> extent = numNodes + haloSize + haloSize;
            constexpr alpaka::Vec<alpaka::DimInt<3u>, Idx> elemPerThread{1, 1, 1};

            auto bufDevice = alpaka::allocBuf<T_data, Idx>(devAcc, extent);
        }
        else if constexpr (DIM==2){
            // order Y X
            const alpaka::Vec<alpaka::DimInt<2u>, Idx> numNodes{blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            const alpaka::Vec<alpaka::DimInt<2u>, Idx> haloSize{blockGrid.getDs()[1], blockGrid.getDs()[0]};
            const alpaka::Vec<alpaka::DimInt<2u>, Idx> extent = numNodes + haloSize + haloSize;
            constexpr alpaka::Vec<alpaka::DimInt<2u>, Idx> elemPerThread{1, 1};
        }
        else if constexpr (DIM==1){
            // order X
            const alpaka::Vec<alpaka::DimInt<1u>, Idx> numNodes{blockGrid.getNlocalNoGuards()[0]};
            const alpaka::Vec<alpaka::DimInt<1u>, Idx> haloSize{blockGrid.getDs()[0]};
            const alpaka::Vec<alpaka::DimInt<1u>, Idx> extent = numNodes + haloSize + haloSize;
            constexpr alpaka::Vec<alpaka::DimInt<1u>, Idx> elemPerThread{1};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }

    }

    private:

};