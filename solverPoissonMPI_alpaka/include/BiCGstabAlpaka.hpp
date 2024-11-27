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
#include "kernelsAlpakaBiCGstab.hpp"


template <int DIM, typename T_data, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class BiCGstabAlpaka : public IterativeSolverBase<DIM,T_data,maxIteration>{
    public:

    BiCGstabAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                    CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBase<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI),
        alpakaHelper_(alpakaHelper),
        preconditioner(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper)
    {
        toll_ = static_cast<T_data>(tolerance) * tollScalingFactor;
        pk = new T_data[this->ntotlocal_guards_];
        rk = new T_data[this->ntotlocal_guards_];
        r0 = new T_data[this->ntotlocal_guards_];
        Mpk = new T_data[this->ntotlocal_guards_];
        AMpk = new T_data[this->ntotlocal_guards_];
        zk = new T_data[this->ntotlocal_guards_];
        Azk = new T_data[this->ntotlocal_guards_];

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(r0, r0 + this->ntotlocal_guards_, 0);
        std::fill(Mpk, Mpk + this->ntotlocal_guards_, 0);
        std::fill(AMpk, AMpk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);
        std::fill(Azk, Azk + this->ntotlocal_guards_, 0);
        
    }

    ~BiCGstabAlpaka()
    {
        delete[] pk;
        delete[] rk;
        delete[] r0;
        delete[] Mpk;
        delete[] AMpk;
        delete[] zk;
        delete[] Azk;
        
    }


    
    void operator()(T_data fieldX[], T_data fieldB[], MatrixFreeOperatorA<DIM,T_data>& operatorA)
    {
        int i,j,k,indx;
        int iter=0;

        std::fill(pk, pk + this->ntotlocal_guards_, 0);
        std::fill(rk, rk + this->ntotlocal_guards_, 0);
        std::fill(r0, r0 + this->ntotlocal_guards_, 0);
        std::fill(Mpk, Mpk + this->ntotlocal_guards_, 0);
        std::fill(AMpk, AMpk + this->ntotlocal_guards_, 0);
        std::fill(zk, zk + this->ntotlocal_guards_, 0);
        std::fill(Azk, Azk + this->ntotlocal_guards_, 0);


        alpaka::Queue<Acc, alpaka::Blocking> queue{this->alpakaHelper_.devAcc_};

        auto fieldXDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto fieldXMdSpan = alpaka::experimental::getMdSpan(fieldXDev);
        auto pkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto pkMdSpan = alpaka::experimental::getMdSpan(pkDev);
        auto rkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto rkMdSpan = alpaka::experimental::getMdSpan(rkDev);
        auto r0Dev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto r0MdSpan = alpaka::experimental::getMdSpan(r0Dev);
        auto MpkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto MpkMdSpan = alpaka::experimental::getMdSpan(MpkDev);
        auto AMpkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto AMpkMdSpan = alpaka::experimental::getMdSpan(AMpkDev);
        auto zkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto zkMdSpan = alpaka::experimental::getMdSpan(zkDev);
        auto AzkDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto AzkMdSpan = alpaka::experimental::getMdSpan(AzkDev);

        auto pitchBuffAcc{alpaka::getPitchesInBytes(pkDev)};
        for(int i=0; i<3; i++)
        {
            pitchBuffAcc[i] /= sizeof(T_data);
        }
        const Idx stride_j_alpaka = pitchBuffAcc[1];
        const Idx stride_k_alpaka = pitchBuffAcc[0];
        int ntotBuffAcc = stride_k_alpaka * this->alpakaHelper_.extent_[0];

        this-> communicatorMPI_.setStridesAlpaka(stride_j_alpaka,stride_k_alpaka);

        const alpaka::Vec<Dim, Idx> extent1D = {1,1,1};
        auto dotPorductHost1 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto dotPorductDev1 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrDotPorductHost1= getPtrNative(dotPorductHost1);
        T_data* const ptrDotPorductDev1{std::data(dotPorductDev1)};

        auto dotPorductHost2 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto dotPorductDev2 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrDotPorductHost2= getPtrNative(dotPorductHost2);
        T_data* const ptrDotPorductDev2{std::data(dotPorductDev2)};
        
        
        
        const alpaka::Vec<Dim, Idx> extent1D2 = {1,1,2};
        auto dotPorductHost2D = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D2);
        auto dotPorductDev2D = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D2);
        auto* const ptrDotPorductHost2D= getPtrNative(dotPorductHost2D);
        T_data* const ptrDotPorductDev2D{std::data(dotPorductDev2D)};
        

        /*
        const alpaka::Vec<Dim, Idx> extent1D = {1,1,1};
        auto dotPorductHost = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto dotPorductDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrDotPorductHost= getPtrNative(dotPorductHost);
        T_data* const ptrDotPorductDev{std::data(dotPorductDev)};

        
        auto alphakHost = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto alphakDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrAlphakHost= getPtrNative(alphakHost);
        T_data* const ptrAlphakDev{std::data(alphakDev)};

        auto betakHost = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto betakDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrBetakHost= getPtrNative(betakHost);
        T_data* const ptrBetakDev{std::data(betakDev)};

        auto omegakHost = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto omegakDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrOmegakHost= getPtrNative(omegakHost);
        T_data* const ptrOmegakDev{std::data(omegakDev)};
        */

        const int imin=this->indexLimitsSolver_[0];
        const int imax=this->indexLimitsSolver_[1];
        const int jmin=this->indexLimitsSolver_[2];
        const int jmax=this->indexLimitsSolver_[3];
        const int kmin=this->indexLimitsSolver_[4];
        const int kmax=this->indexLimitsSolver_[5];
            
        T_data alphak=1;
        T_data betak=1;
        T_data omegak=1;
        T_data pSum1=0.0;
        T_data pSum2=0.0;
        T_data totSum1=0.0;
        T_data totSum2=0.0;
        T_data rho0=1;
        T_data rho1=1;


        if (isMainLoop && communicationON)
        {
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        if constexpr(isMainLoop)
        {
            this-> template resetNeumanBCs<isMainLoop,true>(fieldX);
        }
        else 
        {
            std::fill(fieldX, fieldX + this->ntotlocal_guards_, 0);
        }
        this-> template normalizeProblemToFieldBNorm<isMainLoop,communicationON>(fieldX,fieldB);

        if constexpr(isMainLoop)
        {
            if(this->my_rank_==0)
            {    std::cout<<"Debug in BiCGSTAB Alpaka START " << " main loop "<< isMainLoop << " globalLocation " << this->globalLocation_[0] << " "<< this->globalLocation_[1] <<" "<< this->globalLocation_[2] <<
                " indexLimitsData "<< this->indexLimitsData_[0]<< " "<<this->indexLimitsData_[1] <<" "<< this->indexLimitsData_[2] << " " <<this->indexLimitsData_[3] << " "<< this->indexLimitsData_[4] << " "<< this->indexLimitsData_[5] <<
                " indexLimitsSolver "<< this->indexLimitsSolver_[0]<< " "<<this->indexLimitsSolver_[1] <<" "<< this->indexLimitsSolver_[2] << " " <<this->indexLimitsSolver_[3] << " "<< this->indexLimitsSolver_[4] << " "<< this->indexLimitsSolver_[5] << 
                " norm fieldB " <<this->normFieldB_ <<std::endl;
            }
        }

        // r0= b - A(x0)
        this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,rk,operatorA);
        if constexpr (isMainLoop && trackErrorFromIterationHistory)
        {
            this->errorFromIterationHistory_[0]=this->errorComputeOperator_;
        }
        if(this->errorComputeOperator_ < toll_)
        {   
            this->numIterationFinal_ = 0;
            return;
        }

        //p0=r0
        std::memcpy(pk, rk, this->ntotlocal_guards_* sizeof(T_data));
        std::memcpy(r0, rk, this->ntotlocal_guards_* sizeof(T_data));

        auto fieldXHostView = createView(this->alpakaHelper_.devHost_, fieldX, this->alpakaHelper_.extent_);
        auto pkHostView = createView(this->alpakaHelper_.devHost_, pk, this->alpakaHelper_.extent_);
        auto rkHostView = createView(this->alpakaHelper_.devHost_, rk, this->alpakaHelper_.extent_);
        auto r0HostView = createView(this->alpakaHelper_.devHost_, r0, this->alpakaHelper_.extent_);
        auto MpkHostView = createView(this->alpakaHelper_.devHost_, Mpk, this->alpakaHelper_.extent_);

        memcpy(queue, fieldXDev, fieldXHostView, this->alpakaHelper_.extent_);
        memcpy(queue, pkDev, pkHostView, this->alpakaHelper_.extent_);
        memcpy(queue, rkDev, rkHostView, this->alpakaHelper_.extent_);
        memcpy(queue, r0Dev, r0HostView, this->alpakaHelper_.extent_);
        memcpy(queue, MpkDev, MpkHostView, this->alpakaHelper_.extent_);
        memcpy(queue, AMpkDev, MpkHostView, this->alpakaHelper_.extent_);
        memcpy(queue, zkDev, MpkHostView, this->alpakaHelper_.extent_);
        memcpy(queue, AzkDev, MpkHostView, this->alpakaHelper_.extent_);

        InitializeBufferKernel<3> initBufferKernel;
        alpaka::KernelCfg<Acc> const cfgExtent = {this->alpakaHelper_.extent_, this->alpakaHelper_.elemPerThread_};
        auto workDivExtent0 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, initBufferKernel, pkMdSpan, this->blockGrid_.getMyrank(), stride_j_alpaka,stride_k_alpaka);
        DotProductKernel<DIM,T_data>  dotProductKernel;
        auto workDivExtentDotProduct = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, dotProductKernel, r0MdSpan, r0MdSpan, ptrDotPorductDev1, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        StencilKernel<DIM, T_data> stencilKernel;
        auto workDivExtentStencil = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, stencilKernel, AzkMdSpan, zkMdSpan, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_);
        UpdateFieldWith1FieldKernel<DIM, T_data> updateFieldWith1FieldKernel;
        auto workDivExtentUpdateField1 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, updateFieldWith1FieldKernel, rkMdSpan, AMpkMdSpan, alphak, betak, this->alpakaHelper_.indexLimitsSolverAlpaka_);
        UpdateFieldWith2FieldsKernel<DIM, T_data> updateFieldWith2FieldsKernel;
        auto workDivExtentUpdateField2 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, updateFieldWith2FieldsKernel, pkMdSpan, rkMdSpan, AMpkMdSpan, alphak, betak, omegak, this->alpakaHelper_.indexLimitsSolverAlpaka_);

        BiCGstab1Kernel<DIM, T_data> biCGstab1Kernel;
        auto workDivExtentKernel1 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, biCGstab1Kernel, AMpkMdSpan, MpkMdSpan, r0MdSpan, ptrDotPorductDev1, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
        BiCGstab2Kernel<DIM, T_data> biCGstab2Kernel;
        auto workDivExtentKernel2 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, biCGstab2Kernel, AzkMdSpan, zkMdSpan, rkMdSpan, ptrDotPorductDev2D,
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);                                  
        BiCGstab3Kernel<DIM, T_data> biCGstab3Kernel;
        auto workDivExtentKernel3 = alpaka::getValidWorkDiv(cfgExtent, this->alpakaHelper_.devAcc_, biCGstab3Kernel, rkMdSpan, AzkMdSpan, r0MdSpan, ptrDotPorductDev2D, omegak,
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);                             

        std::cout << "Rank " << this->blockGrid_.getMyrank() << "\n extent" << this->alpakaHelper_.extent_  << 
                    "\n pitches" << " "<< pitchBuffAcc << " stride_j "<<stride_j_alpaka << " stride_k " << stride_k_alpaka << 
                    "\n workDivExtentDotProduct:\n\t" << workDivExtentDotProduct << 
                    "\n workDivExtentStencil:\n\t" << workDivExtentStencil << 
                    "\n workDivExtentUpdateField1:\n\t" << workDivExtentUpdateField1 << 
                    "\n workDivExtentUpdateField2:\n\t" << workDivExtentUpdateField2 << std::endl;

        rho0=1;
        rho1=rho0;
        auto startSolver = std::chrono::high_resolution_clock::now();
        
        while(iter<maxIteration)
        {
            preconditioner(MpkDev,pkDev);
            //memcpy(queue, MpkDev, pkDev, this->alpakaHelper_.extent_);
            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<true>(getPtrNative(MpkDev));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            // to do Add Neuman BCs kernels
            //this-> template resetNeumanBCs<isMainLoop,false>(Mpk);
            /*
            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        AMpk[indx] = operatorA(i,j,k,Mpk);
                        pSum2 += r0[indx] * AMpk[indx];
                    }
                }
            }
            */
            
            //alpaka::exec<Acc>(queue, workDivExtentStencil, stencilKernel, AMpkMdSpan, MpkMdSpan, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_);
            //ptrDotPorductHost[0]=0;
            //memcpy(queue, dotPorductDev, dotPorductHost, extent1D);
            //alpaka::exec<Acc>(queue, workDivExtentDotProduct, dotProductKernel, r0MdSpan, AMpkMdSpan, ptrDotPorductDev, this->alpakaHelper_.indexLimitsSolverAlpaka_);

            alpaka::exec<Acc>(queue, workDivExtentKernel1, biCGstab1Kernel, AMpkMdSpan, MpkMdSpan, r0MdSpan, ptrDotPorductDev1, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            
            memcpy(queue, dotPorductHost1, dotPorductDev1, extent1D);
            pSum2 = ptrDotPorductHost1[0];
            if constexpr(communicationON)
            {
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                alphak=rho0/totSum2;
            }
            else
            {
                alphak=rho0/pSum2;
            }
            

            // sk = rk - alphak Apk
            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        rk[indx] = rk[indx] - alphak * AMpk[indx];
                    }
                }
            }
            */

            alpaka::exec<Acc>(queue, workDivExtentUpdateField1, updateFieldWith1FieldKernel, rkMdSpan, AMpkMdSpan, 1, -alphak, this->alpakaHelper_.indexLimitsSolverAlpaka_);

            //std::fill(zk, zk + this->ntotlocal_guards_, 0);
            preconditioner(zkDev,rkDev);
            //memcpy(queue, zkDev, rkDev, this->alpakaHelper_.extent_);
            if constexpr (communicationON)
            {
                this->communicatorMPI_.template operator()<true>(getPtrNative(zkDev));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            //this-> template resetNeumanBCs<isMainLoop,false>(zk);
            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        Azk[indx] = operatorA(i,j,k,zk);
                    }
                }
            }
            */
            //alpaka::exec<Acc>(queue, workDivExtentStencil, stencilKernel, AzkMdSpan, zkMdSpan, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_);
            /*
            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        Azk[indx] = operatorA(i,j,k,zk);
                        pSum1 +=  rk[indx] * Azk[indx];
                        pSum2 += Azk[indx] * Azk[indx];
                    }
                }
            }
            */
            /*
            alpaka::exec<Acc>(queue, workDivExtentDotProduct, dotProductKernel, rkMdSpan,AzkMdSpan,ptrDotPorductDev, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            memcpy(queue, dotPorductHost, dotPorductDev, extent1D2);
            pSum1 = ptrDotPorductHost[0];
            alpaka::exec<Acc>(queue, workDivExtentDotProduct, dotProductKernel, AzkMdSpan,AzkMdSpan,ptrDotPorductDev, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            memcpy(queue, dotPorductHost, dotPorductDev, extent1D2);
            pSum2 = ptrDotPorductHost[0];
            */
            alpaka::exec<Acc>(queue, workDivExtentKernel2, biCGstab2Kernel, AzkMdSpan, zkMdSpan, rkMdSpan, ptrDotPorductDev2D, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);                        
            //memcpy(queue, dotPorductHost1, dotPorductDev1, extent1D);
            //memcpy(queue, dotPorductHost2, dotPorductDev2, extent1D);
            //pSum1 = dotPorductHost1[0];
            //pSum2 = dotPorductHost1[0];
            memcpy(queue, dotPorductHost2D, dotPorductDev2D, extent1D2);
            pSum1 = *ptrDotPorductHost2D;
            pSum2 = *(ptrDotPorductHost2D+1);
            if  constexpr (communicationON)
            {   
                MPI_Allreduce(&pSum1, &totSum1, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                omegak=totSum1/totSum2;  
            }
            else
            {
                omegak=pSum1/pSum2;
            }

            /*
            for(i=0; i< this->ntotlocal_guards_; i++)
            {
                fieldX[i] = fieldX[i] + alphak*Mpk[i] + omegak*zk[i];
            }
            */
            alpaka::exec<Acc>(queue, workDivExtentUpdateField2, updateFieldWith2FieldsKernel, fieldXMdSpan, MpkMdSpan, zkMdSpan, 1, alphak, omegak, this->alpakaHelper_.indexLimitsSolverAlpaka_);


            /*
            pSum1=0.0;
            pSum2=0.0;
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin;j<jmax; j++)
                {
                    for(i=imin;i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        rk[indx] = rk[indx] - omegak *  Azk[indx];
                        pSum1 += r0[indx] * rk[indx];
                        pSum2 += rk[indx] * rk[indx];
                    }
                }
            }
            */
            alpaka::exec<Acc>(queue, workDivExtentKernel3, biCGstab3Kernel, rkMdSpan, AzkMdSpan, r0MdSpan, ptrDotPorductDev2D, omegak,
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);

            memcpy(queue, dotPorductHost2D, dotPorductDev2D, extent1D2);
            pSum1 = *ptrDotPorductHost2D;
            pSum2 = *(ptrDotPorductHost2D+1);
            
            /*
            alpaka::exec<Acc>(queue, workDivExtentUpdateField1, updateFieldWith1FieldKernel, rkMdSpan, AzkMdSpan, 1, -omegak, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            alpaka::exec<Acc>(queue, workDivExtentDotProduct, dotProductKernel, r0MdSpan, rkMdSpan,ptrDotPorductDev1, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            memcpy(queue, dotPorductHost1, dotPorductDev1, extent1D);
            pSum1 = ptrDotPorductHost1[0];
            alpaka::exec<Acc>(queue, workDivExtentDotProduct, dotProductKernel, rkMdSpan, rkMdSpan,ptrDotPorductDev1, this->alpakaHelper_.indexLimitsSolverAlpaka_);
            memcpy(queue, dotPorductHost1, dotPorductDev1, extent1D);
            pSum2 = ptrDotPorductHost1[0];
            */
            if constexpr (communicationON)
            {
                MPI_Allreduce(&pSum1, &rho1, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&pSum2, &totSum2, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                this->errorFromIteration_ = std::sqrt(totSum2);
            }
            else
            {
                this->errorFromIteration_ = std::sqrt(pSum2);
                rho1=pSum1;
            }
            betak = rho1/rho0 * alphak / omegak;
            rho0=rho1;

            // pk+1 = rk+1 + beta*(pk - omeagak A(pk) )
            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {   
                        indx = i + this->stride_j_*j + this->stride_k_*k;
                        pk[indx] = rk[indx] + betak * (pk[indx] - omegak * AMpk[indx] );
                    }
                }
            }
            */
            alpaka::exec<Acc>(queue, workDivExtentUpdateField2, updateFieldWith2FieldsKernel, pkMdSpan, rkMdSpan, AMpkMdSpan, betak, 1, -betak*omegak, this->alpakaHelper_.indexLimitsSolverAlpaka_);


            iter++;
            if constexpr (isMainLoop)
            {
                if constexpr (trackErrorFromIterationHistory)
                {
                    this->errorFromIterationHistory_[iter]=this->errorFromIteration_;
                }
                
                if(this->my_rank_==0 && iter%10==0)
                {
                    std::cout<<" Debug in BiCGSTAB Alpaka iter "<< iter << " alpha "<< alphak << " omega " << omegak << " rho0 "<< rho0 <<" error "<< this->errorFromIteration_  <<std::endl;
                }
            }
            if(this->errorFromIteration_ < toll_)
            {
                break;
            }
        }

        memcpy(queue, fieldXHostView, fieldXDev, this->alpakaHelper_.extent_);
        alpaka::wait(queue);

        this->numIterationFinal_ = iter;
        if constexpr(communicationON)
        {   
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        //this-> template resetNeumanBCs<isMainLoop,true>(fieldX);
        auto endSolver = std::chrono::high_resolution_clock::now();
        this->durationSolver_ = endSolver - startSolver;

        if constexpr (isMainLoop)
        {
            this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,rk,operatorA);
        }

        for(i=0; i < this->ntotlocal_guards_; i++)
        {
            fieldX[i] *= this->normFieldB_;
            fieldB[i] *= this->normFieldB_;
        }
        this->normFieldB_=1;

        if constexpr(communicationON)
        {   
            this->communicatorMPI_(fieldX);
            this->communicatorMPI_.waitAllandCheckRcv();
        }
    }

    private:
    const AlpakaHelper<DIM,T_data>& alpakaHelper_;
    T_Preconditioner preconditioner;
    T_data toll_;

    T_data* pk;
    T_data* rk;
    T_data* r0;
    T_data* Mpk;
    T_data* AMpk;
    T_data* zk;
    T_data* Azk;
};