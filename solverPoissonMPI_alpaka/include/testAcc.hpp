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
#include "kernelsAlpaka.hpp"

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

        alpaka::Queue<Acc, alpaka::NonBlocking> queue{devAcc};

        alpaka::Vec<alpaka::DimInt<DIM>, Idx> tmpNumNodes;
        alpaka::Vec<alpaka::DimInt<DIM>, Idx> tmpHaloSize;
        alpaka::Vec<alpaka::DimInt<DIM>, Idx> tmpElemPerThread;

        if constexpr (DIM==3){
            // order Z Y X
            //tmpNumNodes = {blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0], blockGrid.getNlocalNoGuards()[2]};
            //tmpHaloSize = {blockGrid.getGuards()[1], blockGrid.getGuards()[0],blockGrid.getGuards()[2]};
            tmpNumNodes = {blockGrid.getNlocalNoGuards()[2], blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {blockGrid.getGuards()[2], blockGrid.getGuards()[1],blockGrid.getGuards()[0]};
            tmpElemPerThread = {1,1,1};           
        }
        else if constexpr (DIM==2){
            // order Y X
            tmpNumNodes = {blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {blockGrid.getGuards()[1], blockGrid.getGuards()[0]};
            tmpElemPerThread = {1,1};
        }
        else if constexpr (DIM==1){
            // order X
            tmpNumNodes = {blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {blockGrid.getGuards()[0]};
            tmpElemPerThread = {1};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }

        const alpaka::Vec<alpaka::DimInt<DIM>, Idx> numNodes = tmpNumNodes;
        const alpaka::Vec<alpaka::DimInt<DIM>, Idx> haloSize = tmpHaloSize;
        const alpaka::Vec<alpaka::DimInt<DIM>, Idx> extent = numNodes + haloSize + haloSize;
        const alpaka::Vec<alpaka::DimInt<DIM>, Idx> elemPerThread = tmpElemPerThread;       
        auto bufDevice = alpaka::allocBuf<T_data, Idx>(devAcc, extent);
        auto bufMdSpan = alpaka::experimental::getMdSpan(bufDevice);
        auto pitchBuffAcc{alpaka::getPitchesInBytes(bufDevice)};

        for(int i=0; i<DIM; i++)
        {
            pitchBuffAcc[i] /= sizeof(T_data);
        }

        int stride_j;
        int stride_k;
        int ntotBuffAcc;
        if constexpr(DIM==3)
        {
            stride_j=pitchBuffAcc[1];
            stride_k=pitchBuffAcc[0];
            ntotBuffAcc=stride_k*extent[0];
        }
        else if constexpr (DIM==2)
        {
            stride_j=pitchBuffAcc[0];
            stride_k=pitchBuffAcc[0]*extent[0];
            ntotBuffAcc=stride_k;
        }
        else
        {
            stride_j=pitchBuffAcc[0];
            stride_k=1;
        }
        this-> communicatorMPI_.setStridesAlpaka(stride_j,stride_k);
        
        auto hostView = createView(devHost, fieldX, extent);
        
        memcpy(queue, bufDevice, hostView, extent);

        InitializeBufferKernel<DIM> initBufferKernel;
        
        alpaka::KernelCfg<Acc> const cfgExtent = {extent, elemPerThread};
        auto workDivExtent = alpaka::getValidWorkDiv(cfgExtent, devAcc, initBufferKernel, bufMdSpan, blockGrid.getMyrank(), stride_j,stride_k);
        
        std::cout << "Rank " << blockGrid.getMyrank() << "\n extent" << extent << "\n numNodes" << numNodes << "\n halo" << 
                    haloSize << " tmphalo "<< tmpHaloSize<< " blockGrid.getGuards() " << blockGrid.getGuards()[0]<< " "<< blockGrid.getGuards()[1] << " " << blockGrid.getGuards()[2] <<
                    "\n pitches" << " "<< pitchBuffAcc<< " " << pitchBuffAcc[0] << " "<< pitchBuffAcc[1] <<" "<< pitchBuffAcc[2] << "\n Workdivision[workDivExtent]:\n\t" << workDivExtent << std::endl;
        //std::experimental::mdspan<T_data,extent> 

        alpaka::exec<Acc>(queue, workDivExtent, initBufferKernel, bufMdSpan, blockGrid.getMyrank(),stride_j,stride_k);
        alpaka::wait(queue);
        
        auto* ptr= getPtrNative(bufDevice);
        /*
        for(int i=0; i< ntotBuffAcc; i++)
        {
            std::cout<< "Indx "<< i << " val " << ptr[i] <<std::endl;
        */


        //CommunicatorMPI communicatorMPIAlpaka(blockGrid,pitchBuffAcc[0],pitchBuffAcc[0]*extent[0]);

        //communicatorMPIAlpaka(getPtrNative(bufDevice));
        //communicatorMPIAlpaka.waitAllandCheckRcv();
        
        //this-> communicatorMPI_.setStridesAlpaka(pitchBuffAcc[0],pitchBuffAcc[0]*extent[0]);       
        this->communicatorMPI_.template operator()<true>(getPtrNative(bufDevice));
        this->communicatorMPI_.waitAllandCheckRcv();

        memcpy(queue, hostView,bufDevice, extent);

    }

    private:

};


template <int DIM, typename T_data, int maxIteration>
class TestAcc2 : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    TestAcc2(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, CommunicatorMPI<DIM,T_data>& communicatorMPI,const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI),
        alpakaHelper_(alpakaHelper)
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

        alpaka::Queue<Acc, alpaka::Blocking> queue{devAcc};

        alpaka::Vec<Dim, Idx> tmpNumNodes;
        alpaka::Vec<Dim, Idx> tmpHaloSize;
        alpaka::Vec<Dim, Idx> tmpElemPerThread;

        if constexpr (DIM==3){
            // order Z Y X
            //tmpNumNodes = {blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0], blockGrid.getNlocalNoGuards()[2]};
            //tmpHaloSize = {blockGrid.getGuards()[1], blockGrid.getGuards()[0],blockGrid.getGuards()[2]};
            tmpNumNodes = {blockGrid.getNlocalNoGuards()[2], blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {blockGrid.getGuards()[2], blockGrid.getGuards()[1],blockGrid.getGuards()[0]};
            tmpElemPerThread = {1,1,1};           
        }
        else if constexpr (DIM==2){
            // order Y X
            tmpNumNodes = {1,blockGrid.getNlocalNoGuards()[1], blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {0,blockGrid.getGuards()[1], blockGrid.getGuards()[0]};
            tmpElemPerThread = {1,1,1};
        }
        else if constexpr (DIM==1){
            // order X
            tmpNumNodes = {1,1,blockGrid.getNlocalNoGuards()[0]};
            tmpHaloSize = {0,0,blockGrid.getGuards()[0]};
            tmpElemPerThread = {1,1,1};
        }
        else
        {
            std::cerr << "Error: DIM is not valid, DIM = "<< DIM << std::endl;
            exit(-1); // Exit with failure status
        }

        const alpaka::Vec<Dim, Idx> numNodes = tmpNumNodes;
        const alpaka::Vec<Dim, Idx> haloSize = tmpHaloSize;
        const alpaka::Vec<Dim, Idx> extent = numNodes + haloSize + haloSize;
        const alpaka::Vec<Dim, Idx> extent1D = {1,1,1};
        const alpaka::Vec<Dim, Idx> elemPerThread = tmpElemPerThread;       
        auto bufDevice = alpaka::allocBuf<T_data, Idx>(devAcc, extent);
        auto bufMdSpan = alpaka::experimental::getMdSpan(bufDevice);
        auto pitchBuffAcc{alpaka::getPitchesInBytes(bufDevice)};

        for(int i=0; i<3; i++)
        {
            pitchBuffAcc[i] /= sizeof(T_data);
        }

        int stride_j;
        int stride_k;
        int ntotBuffAcc;
        stride_j=pitchBuffAcc[1];
        stride_k=pitchBuffAcc[0];
        ntotBuffAcc=stride_k*extent[0];
        this-> communicatorMPI_.setStridesAlpaka(stride_j,stride_k);
        
        auto hostView = createView(devHost, fieldX, extent);
        
        memcpy(queue, bufDevice, hostView, extent);

        InitializeBufferKernel<3> initBufferKernel;
        
        alpaka::KernelCfg<Acc> const cfgExtent = {extent, elemPerThread};
        auto workDivExtent = alpaka::getValidWorkDiv(cfgExtent, devAcc, initBufferKernel, bufMdSpan, blockGrid.getMyrank(), stride_j,stride_k);
        
        std::cout << "Rank " << blockGrid.getMyrank() << "\n extent" << extent << "\n numNodes" << numNodes << "\n halo" << 
                    haloSize << " tmphalo "<< tmpHaloSize<< " blockGrid.getGuards() " << blockGrid.getGuards()[0]<< " "<< blockGrid.getGuards()[1] << " " << blockGrid.getGuards()[2] <<
                    "\n pitches" << " "<< pitchBuffAcc<< " " << pitchBuffAcc[0] << " "<< pitchBuffAcc[1] <<" "<< pitchBuffAcc[2] << " stride_j "<<stride_j << " stride_k " << stride_k << 
                    "\n Workdivision[workDivExtent]:\n\t" << workDivExtent << std::endl;
        //std::experimental::mdspan<T_data,extent> 

        alpaka::exec<Acc>(queue, workDivExtent, initBufferKernel, bufMdSpan, blockGrid.getMyrank(),stride_j,stride_k);
        //alpaka::wait(queue);

        T_data globalSum=0;
        T_data* const ptrGlobalSum=&globalSum;
        //auto globalSumHostView = createView(devHost, ptrGlobalSum, extent1D);
        
        auto globalSumHost = alpaka::allocBuf<T_data, Idx>(devHost, extent1D);
        auto globalSumDev = alpaka::allocBuf<T_data, Idx>(devAcc, extent1D);
        auto* const ptrGlobalSumHost= getPtrNative(globalSumHost);
        ptrGlobalSumHost[0]=1000;
        T_data* const ptrGlobalSumDev{std::data(globalSumDev)};
        memcpy(queue, globalSumDev, globalSumHost, extent1D);


        DotProductKernel<DIM,T_data>  dotProductKernel;
        auto workDivExtentDot = alpaka::getValidWorkDiv(cfgExtent, devAcc, dotProductKernel, bufMdSpan, bufMdSpan, ptrGlobalSumDev, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        alpaka::exec<Acc>(queue, workDivExtentDot, dotProductKernel, bufMdSpan,bufMdSpan,ptrGlobalSumDev, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        alpaka::wait(queue);
        memcpy(queue, globalSumHost, globalSumDev, extent1D);

        std::cout<< "Rank "<< blockGrid.getMyrank() << " indexlimitsSolverAlpaka "<< this->alpakaHelper_.indexLimitsSolverAlpaka_ << " Dot product "<< ptrGlobalSumHost[0] <<std::endl;
        
        auto* ptr= getPtrNative(bufDevice);
        /*
        for(int i=0; i< ntotBuffAcc; i++)
        {
            std::cout<< "Indx "<< i << " val " << ptr[i] <<std::endl;
        */


        //CommunicatorMPI communicatorMPIAlpaka(blockGrid,pitchBuffAcc[0],pitchBuffAcc[0]*extent[0]);

        //communicatorMPIAlpaka(getPtrNative(bufDevice));
        //communicatorMPIAlpaka.waitAllandCheckRcv();
        
        //this-> communicatorMPI_.setStridesAlpaka(pitchBuffAcc[0],pitchBuffAcc[0]*extent[0]);       
        this->communicatorMPI_.template operator()<true>(getPtrNative(bufDevice));
        this->communicatorMPI_.waitAllandCheckRcv();

        auto bufDevice2 = alpaka::allocBuf<T_data, Idx>(devAcc, extent);
        memcpy(queue, bufDevice2,bufDevice, extent);
        memcpy(queue, hostView,bufDevice2, extent);

    }

    private:

    const AlpakaHelper<DIM,T_data>& alpakaHelper_;

};
