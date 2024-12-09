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

        T_data r0 = ds[0]*ds[0];
        T_data r1 = ds[1]*ds[1];
        T_data r2 = ds[2]*ds[2];
        

        if constexpr (DIM==3)
        {
            T_data f0 = -2 * ( 1/r0 + 1/r1 + 1/r2);
            //printf()
            // Z Y X
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = bufData(k,j,i) * f0 +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/r0 + 
                                ( bufData(k,j-1,i) + bufData(k,j+1,i) )/r1 +
                                ( bufData(k-1,j,i) + bufData(k+1,j,i) )/r2 ;
            }
        }
        else if constexpr (DIM==2)
        {
            T_data f0 = -2 * ( 1/r0 + 1/r1);
            // Y X
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = bufData(k,j,i) * f0 +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/r0 + 
                                ( bufData(k,j-1,i) + bufData(k,j+1,i) )/r1;
            }
        }
        else
        {
            T_data f0 = -2 * ( 1/r0 );
            if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
            {
                AbufData(k,j,i) = bufData(k,j,i) * f0 +
                                ( bufData(k,j,i-1) + bufData(k,j,i+1) )/r0;
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




/*



if(i==0 && j==0 && k==0)
        {
            printf("hello v0");
        }
        
        auto const blockThreadExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);
        auto const chunck = blockThreadExtent + haloSize + haloSize;
        //auto& sdata = alpaka::declareSharedVar<T_data[chunck.prod()], __COUNTER__>(acc);
        auto& sdata = alpaka::declareSharedVar<T_data[1024], __COUNTER__>(acc);

        if(i==0 && j==0 && k==0)
        {
            printf("hello");
        }
        
        
        //auto* sdata = alpaka::getDynSharedMem<T_data>(acc);
        // use mdspan for easier set and access the 1D shared memory
        auto sdataMdSpan = std::experimental::mdspan<T_data, std::experimental::dextents<size_t, 3>>(sdata, chunck[0], chunck[1], chunck[2]);

        if( i>=indexLimitsSolver[0] && i<indexLimitsSolver[1] && j>=indexLimitsSolver[2] && j<indexLimitsSolver[3] && k>=indexLimitsSolver[4] && k<indexLimitsSolver[5] )
        {
            auto localIdx = blockThreadIdx + haloSize;
            auto globalIdx = gridThreadIdx;
            sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(k,j,i);
            

            alpaka::Vec<alpaka::DimInt<3>,Idx> shift = {0,0,0}; 

            for(Idx dir=DIM-1; dir >= 0; dir--)
            {
                if(blockThreadIdx[dir] == 0 || gridThreadIdx[dir] == indexLimitsSolver[2*(DIM-1-dir)] )
                {
                    shift = {0,0,0};
                    shift[dir] -= haloSize[dir]; 
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]);
                }
                else if(blockThreadIdx[dir] == ( blockThreadExtent[dir] -1 ) || gridThreadIdx[dir] == indexLimitsSolver[2*(DIM-1-dir)+1])
                {
                    shift = {0,0,0};
                    shift[dir] += haloSize[dir];
                    localIdx = blockThreadIdx + haloSize + shift;
                    globalIdx = gridThreadIdx + shift;
                    sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) = bufDataB(globalIdx[0],globalIdx[1],globalIdx[2]);
                }

            }
            alpaka::syncBlockThreads(acc);
            
            localIdx = blockThreadIdx + haloSize;
            bufDataZ(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2])/theta;
        
            if constexpr (DIM==3)
            {
                fc0 = 2 * rhoCurr / delta * ( 2 - 2 * ( 1/r0 + 1/r1 + 1/r2)/theta );
                f0 = 2 * rhoCurr / delta * (  1/r0 /theta );
                f1 = 2 * rhoCurr / delta * (  1/r1 /theta );
                f2 = 2 * rhoCurr / delta * (  1/r2 /theta );
                //printf()
                // Z Y X fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                bufDataY(k,j,i) = sdataMdSpan(localIdx[0],localIdx[1],localIdx[2]) * fc0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] - 1) + sdataMdSpan(localIdx[0],localIdx[1],localIdx[2] + 1) ) * f0 + 
                                    ( sdataMdSpan(localIdx[0],localIdx[1] - 1,localIdx[2]) + sdataMdSpan(localIdx[0],localIdx[1] + 1,localIdx[2]) ) * f1 +
                                    ( sdataMdSpan(localIdx[0] - 1,localIdx[1],localIdx[2]) + sdataMdSpan(localIdx[0] + 1,localIdx[1],localIdx[2]) ) * f2;  
             }
            else if constexpr (DIM==2)
            {
            }
            else
            {
            }
        }


// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<int DIM, typename T_data, typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev1Kernel<DIM,T_data>, TAcc>
    {
        template<typename TVec, typename... TArgs   >
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev1Kernel<DIM,T_data> const&,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent,
            TArgs&&...)  -> std::size_t
        {
            // Reserve the buffer, buffers size is the number of elements in a block (tile)
            //auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            //return static_cast<std::size_t>(blockThreadExtent.prod() * threadElemExtent.prod()) * sizeof(T_data);
            return static_cast<std::size_t>( 1*sizeof(T_data) );
        }
    };
} // namespace alpaka::trait




// The specialisation used for calculation of dynamic shared memory size
namespace alpaka::trait
{
    //! The trait for getting the size of the block shared dynamic memory for a kernel.
    template<typename TAcc>
    struct BlockSharedMemDynSizeBytes<Chebyshev1Kernel<DIM,T_data>, TAcc>
    {
        template<typename TVec, typename TMdSpan, typename T_data, typename Idx>
        ALPAKA_FN_HOST_ACC static auto getBlockSharedMemDynSizeBytes(
            Chebyshev1Kernel<DIM,T_data> const&  matMulKernel ,
            TVec const& blockThreadExtent, // dimensions of thread per block
            TVec const& threadElemExtent, // dimensions of elements per thread
            TMdSpan bufDataB, 
            TMdSpan bufDataY, 
            TMdSpan bufDataZ, 
            const T_data delta, 
            const T_data theta, 
            const T_data rhoCurr,
            const alpaka::Vec<alpaka::DimInt<6>, Idx> indexLimitsSolver, 
            const alpaka::Vec<alpaka::DimInt<3>, T_data> ds, 
            const alpaka::Vec<alpaka::DimInt<3>, Idx> haloSize) // allocated filter width
        {
            // Reserve the buffer, buffers size is the number of elements in a block (tile)
            auto blockThreadExtentNew = blockThreadExtent + haloSize + haloSize;
            return static_cast<std::size_t>(blockThreadExtentNew.prod() * threadElemExtent.prod()) * sizeof(T_data);
        }
    };
} // namespace alpaka::trait

*/