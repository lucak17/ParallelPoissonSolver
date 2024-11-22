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



template<int DIM, typename T_data> 
struct StencilKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan AbufData, TMdSpan bufData, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver,
                                    const alpaka::Vec<alpaka::DimInt<3>, T_data> ds) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        if constexpr (DIM==3)
        {
            //printf()
            // Z Y X
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = -2*bufData(k,j,i)*( 1/(ds[0]*ds[0]) + 1/(ds[1]*ds[1]) +1/(ds[2]*ds[2]) ) +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/(ds[0]*ds[0]) + 
                                ( bufData(k,j-1,i) + bufData(k,j+1,i) )/(ds[1]*ds[1]) +
                                ( bufData(k-1,j,i) + bufData(k+1,j,i) )/(ds[2]*ds[2]);
            }
        }
        else if constexpr (DIM==2)
        {
            // Y X
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = -2*bufData(k,j,i)*( 1/(ds[0]*ds[0]) + 1/(ds[1]*ds[1]) ) +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/(ds[0]*ds[0]) + 
                                ( bufData(k,j-1,i) + bufData(k,j+1,i) )/(ds[1]*ds[1]);
            }
        }
        else
        {
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = -2*bufData(k,j,i)*( 1/(ds[0]*ds[0]) ) +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/(ds[0]*ds[0]);
            }
        }
    }
};

template<int DIM, typename T_data> 
struct DotProductKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB,  T_data* const globalSum, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        auto const iBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[2];
        auto const jBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[1];
        auto const kBlock = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0];

        T_data local = 0;
        T_data& blockSum = alpaka::declareSharedVar<T_data, __COUNTER__>(acc);


        if(i==0 && j==0 && k==0)
        {
            *globalSum = 0;
        }
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            blockSum = 0;
        }
        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            local = bufDataA(k,j,i) * bufDataB(k,j,i);
        }
        
        alpaka::atomicAdd(acc, &blockSum, local, alpaka::hierarchy::Threads{});
        alpaka::syncBlockThreads(acc);

        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            alpaka::atomicAdd(acc, globalSum, blockSum, alpaka::hierarchy::Blocks{});
        }
    }
};





template<int DIM, typename T_data> 
struct UpdateFieldWith1FieldKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data constA, const T_data constB,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) = constA * bufDataA(k,j,i) + constB * bufDataB(k,j,i);
        }
    }
};



template<int DIM, typename T_data> 
struct UpdateFieldWith2FieldsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, TMdSpan bufDataC, const T_data constA, const T_data constB, const T_data constC,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) = constA * bufDataA(k,j,i) + constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i) ;
        }
    }
};


template<int DIM, typename T_data> 
struct AssignFieldWith1FieldKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, const T_data constB,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i);
        }
    }
};

template<int DIM, typename T_data> 
struct AssignFieldWith2FieldsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, TMdSpan bufDataC, const T_data constB, const T_data constC,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i);
        }
    }
};


template<int DIM, typename T_data> 
struct AssignFieldWith4FieldsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataA, TMdSpan bufDataB, TMdSpan bufDataC, TMdSpan bufDataD, TMdSpan bufDataE, 
                                    const T_data constB, const T_data constC, const T_data constD, const T_data constE,
                                    const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver) const -> void
    {
        // Get indexes
        auto const i = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[2];
        auto const j = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[1];
        auto const k = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];

        // Z Y X
        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            bufDataA(k,j,i) =  constB * bufDataB(k,j,i) + constC * bufDataC(k,j,i) + constD * bufDataD(k,j,i) + constE * bufDataE(k,j,i);
        }
    }
};

