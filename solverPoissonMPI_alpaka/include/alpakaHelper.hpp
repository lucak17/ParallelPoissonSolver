#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once

#include <alpaka/alpaka.hpp>

#include "blockGrid.hpp"
#include "communicationMPI.hpp"

template <int DIM, typename T_data>
class AlpakaHelper {
    public:

    AlpakaHelper(const BlockGrid<DIM,T_data>& blockGrid):
        blockGrid_(blockGrid),
        platformHost_(alpaka::PlatformCpu{}),
        devHost_(alpaka::getDevByIdx(alpaka::PlatformCpu{}, 0)),
        platformAcc_(alpaka::Platform<Acc>{}),
        devAcc_(alpaka::getDevByIdx(alpaka::Platform<Acc>{}, 0)),
        elemPerThread_({1,1,1})
    {
        
        if constexpr (DIM==3){
            // order Z Y X
            numCells_ = {blockGrid.getNlocalNoGuards()[2], blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            haloSize_ = {blockGrid.getGuards()[2], blockGrid.getGuards()[1],blockGrid.getGuards()[0]};        
        }
        else if constexpr (DIM==2){
            // order Y X
            numCells_ = {1,blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            haloSize_ = {0,blockGrid.getGuards()[1], blockGrid.getGuards()[0]};
        }
        else if constexpr (DIM==1){
            // order X
            numCells_ = {1,1,blockGrid.getNlocalNoGuards()[0]};
            haloSize_ = {0,0,blockGrid.getGuards()[0]};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }

        extent_ = numCells_ + haloSize_ + haloSize_;
    }

    private:
    const BlockGrid<DIM,T_data>& blockGrid_;
    const alpaka::PlatformCpu platformHost_;
    const alpaka::Dev<alpaka::PlatformCpu> devHost_;
    const alpaka::Platform<Acc> platformAcc_; 
    const alpaka::Dev<alpaka::Platform<Acc>> devAcc_;
    const alpaka::Vec<Dim, Idx> elemPerThread_;
    alpaka::Vec<Dim, Idx> numCells_;
    alpaka::Vec<Dim, Idx> haloSize_;
    alpaka::Vec<Dim, Idx> extent_;
    
};