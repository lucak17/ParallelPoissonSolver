#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 
#include <cassert>

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
        elemPerThread_({1,1,1}),
        numCells_(setNumCells(blockGrid)),
        haloSize_(setHaloSize(blockGrid)),
        extent_(setNumCells(blockGrid) + setHaloSize(blockGrid) + setHaloSize(blockGrid) ),
        extentNoHalo_(setNumCells(blockGrid)),
        extentSolver_({blockGrid.getIndexLimitsSolver()[5] - blockGrid.getIndexLimitsSolver()[4],blockGrid.getIndexLimitsSolver()[3] - blockGrid.getIndexLimitsSolver()[2],blockGrid.getIndexLimitsSolver()[1] - blockGrid.getIndexLimitsSolver()[0]}),
        numBlocksGrid_(setNumBlocks()),
        indexLimitsSolverAlpaka_({blockGrid.getIndexLimitsSolver()[0],blockGrid.getIndexLimitsSolver()[1],blockGrid.getIndexLimitsSolver()[2],blockGrid.getIndexLimitsSolver()[3],blockGrid.getIndexLimitsSolver()[4],blockGrid.getIndexLimitsSolver()[5]}),
        ds_({blockGrid.getDs()[0],blockGrid.getDs()[1],blockGrid.getDs()[2]})
    {
    }


    const BlockGrid<DIM,T_data>& blockGrid_;
    const alpaka::PlatformCpu platformHost_;
    const alpaka::Dev<alpaka::PlatformCpu> devHost_;
    const alpaka::Platform<Acc> platformAcc_; 
    const alpaka::Dev<alpaka::Platform<Acc>> devAcc_;
    const alpaka::Vec<Dim, Idx> elemPerThread_;
    const alpaka::Vec<Dim, Idx> numCells_;
    const alpaka::Vec<Dim, Idx> haloSize_;
    const alpaka::Vec<Dim, Idx> extent_;
    const alpaka::Vec<Dim, Idx> extentNoHalo_;
    const alpaka::Vec<Dim, Idx> extentSolver_;
    const alpaka::Vec<Dim, Idx> numBlocksGrid_;
    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolverAlpaka_;
    const alpaka::Vec<alpaka::DimInt<3>, T_data> ds_;

    private:

    auto setNumCells(const BlockGrid<DIM,T_data>& blockGrid) const
    {
        alpaka::Vec<Dim, Idx> numCellsTmp;
        if constexpr (DIM==3){
            // order Z Y X
            numCellsTmp = {blockGrid.getNlocalNoGuards()[2], blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};    
        }
        else if constexpr (DIM==2){
            // order Y X
            numCellsTmp = {1,blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
        }
        else if constexpr (DIM==1){
            // order X
            numCellsTmp = {1,1,blockGrid.getNlocalNoGuards()[0]};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }
        return numCellsTmp;
    }


    auto setHaloSize(const BlockGrid<DIM,T_data>& blockGrid) const
    {
        alpaka::Vec<Dim, Idx> haloSizeTmp;
        if constexpr (DIM==3){
            // order Z Y X
            haloSizeTmp = {blockGrid.getGuards()[2], blockGrid.getGuards()[1],blockGrid.getGuards()[0]};        
        }
        else if constexpr (DIM==2){
            // order Y X
            haloSizeTmp = {0,blockGrid.getGuards()[1], blockGrid.getGuards()[0]};
        }
        else if constexpr (DIM==1){
            // order X
            haloSizeTmp = {0,0,blockGrid.getGuards()[0]};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }
        return haloSizeTmp;
    }

    auto setNumBlocks()
    {
        alpaka::Vec<Dim, Idx> numBlocksTmp;
        
        numBlocksTmp[0] = extentNoHalo_[0]/blockExtentFixed[0];
        numBlocksTmp[1] = extentNoHalo_[1]/blockExtentFixed[1];
        numBlocksTmp[2] = extentNoHalo_[2]/blockExtentFixed[2];

        for(int i=0; i<3; i++)
        {
            if (extentNoHalo_[i] != numBlocksTmp[i] * blockExtentFixed[i])
            {
                std::cerr << "Error: extensions in direction "<< i <<" is not divisible by blockExtentFixed" << std::endl;
                exit(-1);
            }
        }
        
        return numBlocksTmp;
    }
    
};