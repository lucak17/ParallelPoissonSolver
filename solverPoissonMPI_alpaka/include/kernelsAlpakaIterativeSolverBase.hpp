#pragma once

#include <alpaka/alpaka.hpp>
#include <cstdio>


template<int DIM, typename T_data> 
struct DotProductKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, const TMdSpan bufDataA, const TMdSpan bufDataB,  T_data* const globalSum, 
                                    const alpaka::Vec<alpaka::DimInt<3>, Idx> offset) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const iBlock = blockThreadIdx[2];
        auto const jBlock = blockThreadIdx[1];
        auto const kBlock = blockThreadIdx[0];

        T_data local = 0;
        T_data& blockSum = alpaka::declareSharedVar<T_data, __COUNTER__>(acc);


        if(gridThreadIdx[0]==0 && gridThreadIdx[1]==0 && gridThreadIdx[2]==0)
        {
            *globalSum = 0.0;
        }
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            blockSum = 0.0;
        }
        // Z Y X
        local = bufDataA(kGrid,jGrid,iGrid) * bufDataB(kGrid,jGrid,iGrid);       
        alpaka::atomicAdd(acc, &blockSum, local, alpaka::hierarchy::Threads{});
        alpaka::syncBlockThreads(acc);

        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            alpaka::atomicAdd(acc, globalSum, blockSum, alpaka::hierarchy::Blocks{});
        }
    }
};



template<int DIM, typename T_data> 
struct DotProductNormKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, const TMdSpan bufDataA, const TMdSpan bufDataB,  T_data* const globalSum, 
                                    const alpaka::Vec<alpaka::DimInt<3>, Idx> offset) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const iBlock = blockThreadIdx[2];
        auto const jBlock = blockThreadIdx[1];
        auto const kBlock = blockThreadIdx[0];

        T_data local = 0;
        T_data& blockSum = alpaka::declareSharedVar<T_data, __COUNTER__>(acc);


        if(gridThreadIdx[0]==0 && gridThreadIdx[1]==0 && gridThreadIdx[2]==0)
        {
            *globalSum = 0.0;
        }
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            blockSum = 0.0;
        }
        // Z Y X
        local = bufDataA(kGrid,jGrid,iGrid) * bufDataA(kGrid,jGrid,iGrid);
        if(local < 0)
            printf("Local < 0 ");       
        alpaka::atomicAdd(acc, &blockSum, local, alpaka::hierarchy::Threads{});
        alpaka::syncBlockThreads(acc);
        if(blockSum < 0)
            printf("blockSum < 0 ");
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            alpaka::atomicAdd(acc, globalSum, blockSum, alpaka::hierarchy::Blocks{});
        }
        alpaka::syncBlockThreads(acc);
        
        
        if(*globalSum < 0)
        {   
            if(iBlock==0 && jBlock==0 && kBlock==0)
            {
                printf("globalSum < 0, globalSum = %.17g - blockSum = %.17g \n", *globalSum, blockSum);
                printf("\n");
            }
        }
    }
};



template<int DIM, typename T_data, bool fieldData, bool negativeDir> 
struct ResetNeumanBCsKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs, const Idx dir, const T_data normFieldB, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsData, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, T_data> origin, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> globalLocation, const alpaka::Vec<alpaka::DimInt<3>, Idx> nlocal_noguards,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const indxData = gridThreadIdxShifted + adjustIdx;
        auto const indxGuard = gridThreadIdxShifted - adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x=origin[2] + (iGrid-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
        const T_data y=origin[1] + (jGrid-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
        const T_data z=origin[0] + (kGrid-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            if constexpr (fieldData)
            {
                if constexpr (negativeDir)
                {
                    bufData(indxGuard[0],indxGuard[1],indxGuard[2]) = bufData(indxData[0],indxData[1],indxData[2]) - 2 * ds[dir] * exactSolutionAndBCs.trueSolutionDdir(x,y,z, 2 - dir) / normFieldB;
                }
                else
                {
                    bufData(indxGuard[0],indxGuard[1],indxGuard[2]) = bufData(indxData[0],indxData[1],indxData[2]) + 2 * ds[dir] * exactSolutionAndBCs.trueSolutionDdir(x,y,z, 2 - dir) / normFieldB;
                }
            }   
            else
            {
                bufData(indxGuard[0],indxGuard[1],indxGuard[2]) = bufData(indxData[0],indxData[1],indxData[2]);
            }
        }

    }
};



template<int DIM, typename T_data, bool negativeDir> 
struct ApplyNeumanBCsFieldBKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs, const Idx dir, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsData, 
                                const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, T_data> origin, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> globalLocation, const alpaka::Vec<alpaka::DimInt<3>, Idx> nlocal_noguards,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        //auto const indxData = gridThreadIdxShifted + adjustIdx;
        //auto const indxGuard = gridThreadIdxShifted - adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x=origin[2] + (iGrid-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
        const T_data y=origin[1] + (jGrid-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
        const T_data z=origin[0] + (kGrid-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            if constexpr (negativeDir)
            {
                bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) + 
                                                                                                    2 * exactSolutionAndBCs.trueSolutionDdir(x,y,z, 2 - dir) / ds[dir];
            }
            else
            {
                bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) -  
                                                                                                    2 * exactSolutionAndBCs.trueSolutionDdir(x,y,z, 2 - dir) / ds[dir];
            }
        }
    }
};


template<int DIM, typename T_data> 
struct ApplyDirichletBCsFieldBKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufDataX, TMdSpan bufDataB, const Idx dir,
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> adjustIdx, const alpaka::Vec<alpaka::DimInt<3>, T_data> ds,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const indxB = gridThreadIdxShifted + adjustIdx;
        //auto const indxGuard = gridThreadIdxShifted - adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            bufDataB(indxB[0],indxB[1],indxB[2]) -= bufDataX(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) / (ds[dir]*ds[dir]);
        }
    }
};



template<int DIM, typename T_data> 
struct ApplyDirichletBCsFromFunctionKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsEdge, const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsData, 
                                const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, T_data> origin, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> globalLocation, const alpaka::Vec<alpaka::DimInt<3>, Idx> nlocal_noguards,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        //auto const indxData = gridThreadIdxShifted + adjustIdx;
        //auto const indxGuard = gridThreadIdxShifted - adjustIdx;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x=origin[2] + (iGrid-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
        const T_data y=origin[1] + (jGrid-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
        const T_data z=origin[0] + (kGrid-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];


        if( iGrid>=indexLimitsEdge[4] && iGrid<indexLimitsEdge[5] && jGrid>=indexLimitsEdge[2] && jGrid<indexLimitsEdge[3] && kGrid>=indexLimitsEdge[0] && kGrid<indexLimitsEdge[1])
        {
            bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = exactSolutionAndBCs.trueSolutionFxyz(x,y,z);
        }
    }
};


template<int DIM, typename T_data> 
struct SetFieldValuefromFunctionKernel
{
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan bufData, const ExactSolutionAndBCs<DIM,T_data> exactSolutionAndBCs, 
                                const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsData, 
                                const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, T_data> origin, 
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> globalLocation, const alpaka::Vec<alpaka::DimInt<3>, Idx> nlocal_noguards,
                                const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize ) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
    
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];

        const T_data x=origin[2] + (iGrid-indexLimitsData[4])*ds[2] + globalLocation[2]*(nlocal_noguards[2])*ds[2];
        const T_data y=origin[1] + (jGrid-indexLimitsData[2])*ds[1] + globalLocation[1]*(nlocal_noguards[1])*ds[1];
        const T_data z=origin[0] + (kGrid-indexLimitsData[0])*ds[0] + globalLocation[0]*(nlocal_noguards[0])*ds[0];


        bufData(gridThreadIdxShifted[0],gridThreadIdxShifted[1],gridThreadIdxShifted[2]) = exactSolutionAndBCs.setFieldB(x,y,z);
    }
};



template<int DIM, typename T_data> 
struct ComputeErrorOperatorAKernel
{
    /*
    for(k=kmin; k<kmax; k++)
    {
        for(j=jmin; j<jmax; j++)
        {
            for(i=imin; i<imax; i++)
            {
                indx = i + this->stride_j_*j + this->stride_k_*k;
                rk[indx] = fieldB[indx] - operatorA(i,j,k,fieldX);
                pSum += rk[indx]*rk[indx];
            }
        }
    }
    */
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, const TMdSpan fieldXMdSpan, const TMdSpan fieldBMdSpan, TMdSpan rkMdSpan, T_data* const globalSum,
                                 const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, const alpaka::Vec<alpaka::DimInt<3>, Idx> offset) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + offset;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        
        auto const blockThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        auto const iBlock = blockThreadIdx[2];
        auto const jBlock = blockThreadIdx[1];
        auto const kBlock = blockThreadIdx[0];

        const T_data r0 = ds[2]*ds[2];
        const T_data r1 = ds[1]*ds[1];
        const T_data r2 = ds[0]*ds[0];
        
        T_data local = 0;
        T_data& blockSum = alpaka::declareSharedVar<T_data, __COUNTER__>(acc);


        if(gridThreadIdx[0]==0 && gridThreadIdx[1]==0 && gridThreadIdx[2]==0)
        {
            *globalSum = 0;
        }
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            blockSum = 0;
        }
                      
        if constexpr (DIM==3)
        {
            const T_data fc0 =  - 2 * ( 1/r0 + 1/r1 + 1/r2 );
            // Z Y X
            rkMdSpan(kGrid,jGrid,iGrid) = fieldBMdSpan(kGrid,jGrid,iGrid) -  fieldXMdSpan(kGrid,jGrid,iGrid) * fc0 -
                            ( fieldXMdSpan(kGrid,jGrid,iGrid-1) + fieldXMdSpan(kGrid,jGrid,iGrid+1) )/r0 - 
                            ( fieldXMdSpan(kGrid,jGrid-1,iGrid) + fieldXMdSpan(kGrid,jGrid+1,iGrid) )/r1 -
                            ( fieldXMdSpan(kGrid-1,jGrid,iGrid) + fieldXMdSpan(kGrid+1,jGrid,iGrid) )/r2 ;
        }
        else if constexpr (DIM==2)
        {
            const T_data fc0 = - 2 * ( 1/r0 + 1/r1 );
            // Z Y X
            rkMdSpan(kGrid,jGrid,iGrid) = fieldBMdSpan(kGrid,jGrid,iGrid) - fieldXMdSpan(kGrid,jGrid,iGrid) * fc0 -
                            ( fieldXMdSpan(kGrid,jGrid,iGrid-1) + fieldXMdSpan(kGrid,jGrid,iGrid+1) )/r0 - 
                            ( fieldXMdSpan(kGrid,jGrid-1,iGrid) + fieldXMdSpan(kGrid,jGrid+1,iGrid) )/r1;
        }
        else
        {
            const T_data fc0 = - 2 * ( 1/r0 );
            // Z Y X
            rkMdSpan(kGrid,jGrid,iGrid) = fieldBMdSpan(kGrid,jGrid,iGrid) - fieldXMdSpan(kGrid,jGrid,iGrid) * fc0 -
                            ( fieldXMdSpan(kGrid,jGrid,iGrid-1) + fieldXMdSpan(kGrid,jGrid,iGrid+1) )/r0;
        }

        local = rkMdSpan(kGrid,jGrid,iGrid) * rkMdSpan(kGrid,jGrid,iGrid);

        alpaka::atomicAdd(acc, &blockSum, local, alpaka::hierarchy::Threads{});
        alpaka::syncBlockThreads(acc);
        if(iBlock==0 && jBlock==0 && kBlock==0)
        {
            alpaka::atomicAdd(acc, globalSum, blockSum, alpaka::hierarchy::Blocks{});
        }
    }
};


template<int DIM, typename T_data> 
struct ResetNormFieldsKernel
{
    /*
    for(i=0; i < this->ntotlocal_guards_; i++)
    {
        fieldX[i] *= this->normFieldB_;
        fieldB[i] *= this->normFieldB_;
    }
    */
    template<typename TAcc, typename TMdSpan>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, TMdSpan fieldXMdSpan, TMdSpan fieldBMdSpan, const T_data normFieldB,
                                  const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize) const -> void
    {
        // Get indexes
        auto const gridThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const gridThreadIdxShifted = gridThreadIdx + haloSize;
        auto const iGrid = gridThreadIdxShifted[2];
        auto const jGrid = gridThreadIdxShifted[1];
        auto const kGrid = gridThreadIdxShifted[0];
        

        fieldXMdSpan(kGrid,jGrid,iGrid) = fieldXMdSpan(kGrid,jGrid,iGrid) * normFieldB;
        fieldBMdSpan(kGrid,jGrid,iGrid) = fieldBMdSpan(kGrid,jGrid,iGrid) * normFieldB;
   }
};