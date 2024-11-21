#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>

template<int DIM> 
struct InitializeBufferKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, int myrank, int stride_j, int stride_k) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);

        if constexpr (DIM==3)
        {
            //printf()
            // Z Y X
            //bufData(gridThreadIdx[0], gridThreadIdx[1], gridThreadIdx[2]) = gridThreadIdx[2] + gridThreadIdx[1]* stride_j + gridThreadIdx[0]*stride_k;
            bufData(gridThreadIdx[0], gridThreadIdx[1], gridThreadIdx[2]) = myrank;
        }
        else if constexpr (DIM==2)
        {
            // Y X
            bufData(gridThreadIdx[0], gridThreadIdx[1]) = gridThreadIdx[1] + gridThreadIdx[0] * stride_j;
        }
        else
        {
            bufData(gridThreadIdx[0]) = myrank;
        }
    }
};

struct InitializeBufferKernel2
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, int myrank) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        
        bufData(gridThreadIdx[0], gridThreadIdx[1],gridThreadIdx[2]) = myrank;
    }
};