#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 
#include <chrono>
#include <iomanip>
#include <ctime>

#pragma once
#include <alpaka/alpaka.hpp>
#include "iterativeSolverBaseAlpaka.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpakaBiCGstab.hpp"


template <int DIM, typename T_data, typename T_data_BiCGStab, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class BiCGstabAlpaka : public IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>{
    public:

    BiCGstabAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                    CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        preconditioner(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        toll_(static_cast<T_data>(tolerance) * tollScalingFactor),
        pkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        rkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        r0Dev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        MpkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        AMpkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        zkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        AzkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        pitchBuffAcc({alpaka::getPitchesInBytes(pkDev)}),
        workDivExtentFixed(alpaka::WorkDivMembers<Dim,Idx>(this->alpakaHelper_.numBlocksGrid_,blockExtentFixed,this->alpakaHelper_.elemPerThread_)),
        workDivExtentKernel1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, BiCGstab1Kernel<DIM, T_data> {}, 
                                                    alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), 
                                                    this->ptrTmp, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentKernel2(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, BiCGstab2KernelSharedMem<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->pkDev),alpaka::experimental::getMdSpan(this->pkDev),alpaka::experimental::getMdSpan(this->pkDev),
                                                    this->ptrTmp,this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentKernel3(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, BiCGstab3Kernel<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), this->ptrTmp, 1.0,
                                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentUpdateField1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, UpdateFieldWith1FieldKernelV2<DIM, T_data>{}, 
                                                    alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), 1.0, 1.0,
                                                    this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.haloSize_)),
        workDivExtentUpdateField2(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, UpdateFieldWith2FieldsKernelV2<DIM, T_data>{},
                                                    alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev), alpaka::experimental::getMdSpan(this->pkDev),
                                                    1.0, 1.0, 1.0,this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.haloSize_))     
    {
        for(int i=0; i<3; i++)
        {
            pitchBuffAcc[i] /= sizeof(T_data);
        }
        stride_j_alpaka = pitchBuffAcc[1];
        stride_k_alpaka = pitchBuffAcc[0];
        //int ntotBuffAcc = stride_k_alpaka * this->alpakaHelper_.extent_[0];
        this-> communicatorMPI_.setStridesAlpaka(stride_j_alpaka,stride_k_alpaka);        
    }

    ~BiCGstabAlpaka()
    {      
    }

    
    void operator()(T_data fieldX[], T_data fieldB[])
    {
        int iter=0;
        T_data alphak=1;
        T_data betak=1;
        T_data omegak=1;
        T_data pSum1=0.0;
        T_data pSum2=0.0;
        T_data totSum1=0.0;
        T_data totSum2=0.0;
        T_data rho0=1;
        T_data rho1=1;

        T_data pSum[2];
        T_data totSum[2];

        constexpr std::uint8_t value0{0}; 
        
        auto rkMdSpan = alpaka::experimental::getMdSpan(rkDev);
        auto r0MdSpan = alpaka::experimental::getMdSpan(r0Dev);
        auto MpkMdSpan = alpaka::experimental::getMdSpan(MpkDev);
        auto AMpkMdSpan = alpaka::experimental::getMdSpan(AMpkDev);
        auto zkMdSpan = alpaka::experimental::getMdSpan(zkDev);
        auto AzkMdSpan = alpaka::experimental::getMdSpan(AzkDev);
        auto pkMdSpan = alpaka::experimental::getMdSpan(pkDev);

        alpaka::memset(this->queueSolverNonBlocking1_, rkDev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, r0Dev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, pkDev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, MpkDev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, AMpkDev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, zkDev, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, AzkDev, value0);

        auto fieldXHostView = createView(this->alpakaHelper_.devHost_, fieldX, this->alpakaHelper_.extent_);
        auto fieldBHostView = createView(this->alpakaHelper_.devHost_, fieldB, this->alpakaHelper_.extent_);

        auto fieldXDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto fieldXMdSpan = alpaka::experimental::getMdSpan(fieldXDev);
        auto fieldBDev = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
        auto fieldBMdSpan = alpaka::experimental::getMdSpan(fieldBDev);

        memcpy(this->queueSolverNonBlocking1_, fieldXDev, fieldXHostView, this->alpakaHelper_.extent_);
        memcpy(this->queueSolverNonBlocking1_, fieldBDev, fieldBHostView, this->alpakaHelper_.extent_);
        alpaka::wait(this->queueSolverNonBlocking1_);
             

        const alpaka::Vec<Dim, Idx> extent1D = {1,1,1};
        auto dotPorductHost1 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D);
        auto dotPorductDev1 = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D);
        auto* const ptrDotPorductHost1= getPtrNative(dotPorductHost1);
        T_data* const ptrDotPorductDev1{std::data(dotPorductDev1)};

        const alpaka::Vec<Dim, Idx> extent1D2 = {1,1,2};
        auto dotPorductHost2D = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devHost_, extent1D2);
        auto dotPorductDev2D = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D2);
        auto* const ptrDotPorductHost2D= getPtrNative(dotPorductHost2D);
        T_data* const ptrDotPorductDev2D{std::data(dotPorductDev2D)};

        auto totSumView = createView(this->alpakaHelper_.devHost_, totSum, extent1D2);
        auto totSumDev2D = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, extent1D2);
        T_data* const ptrTotSumDev2D{std::data(totSumDev2D)};
            
        
        if constexpr (isMainLoop)
        {
            this->setProblemAlpaka(fieldXDev, fieldBDev);
        }
        

        if (isMainLoop && communicationON)
        {
            this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(fieldXDev));
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        if constexpr(isMainLoop)
        {
            this-> template resetNeumanBCsAlpaka<isMainLoop,true>(fieldXDev);
        }
        else 
        {
            alpaka::memset(this->queueSolver_, fieldXDev, value0);
        }

        this-> template normalizeProblemToFieldBNormAlpaka<isMainLoop,communicationON>(fieldXDev,fieldBDev);

        // r0= b - A(x0)
        this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(fieldXDev,fieldBDev,rkDev);

        if constexpr(isMainLoop)
        {
            if(this->my_rank_==0)
            {    std::cout<<"Debug in BiCGSTAB Alpaka START " << " main loop "<< isMainLoop << " globalLocation " << this->globalLocation_[0] << " "<< this->globalLocation_[1] <<" "<< this->globalLocation_[2] <<
                " indexLimitsData "<< this->indexLimitsData_[0]<< " "<<this->indexLimitsData_[1] <<" "<< this->indexLimitsData_[2] << " " <<this->indexLimitsData_[3] << " "<< this->indexLimitsData_[4] << " "<< this->indexLimitsData_[5] <<
                " indexLimitsSolver "<< this->indexLimitsSolver_[0]<< " "<<this->indexLimitsSolver_[1] <<" "<< this->indexLimitsSolver_[2] << " " <<this->indexLimitsSolver_[3] << " "<< this->indexLimitsSolver_[4] << " "<< this->indexLimitsSolver_[5] <<std::endl; 
                std::cout<<"Initial error: " <<  this->errorComputeOperator_ <<" - Norm fieldB " <<this->normFieldB_ <<std::endl;
            }
        }

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
        memcpy(this->queueSolverNonBlocking1_, r0Dev, rkDev, this->alpakaHelper_.extent_);
        memcpy(this->queueSolverNonBlocking1_, pkDev, rkDev, this->alpakaHelper_.extent_);
        alpaka::wait(this->queueSolverNonBlocking1_);

             
        BiCGstab1Kernel<DIM, T_data> biCGstab1Kernel;
        BiCGstab2KernelSharedMem<DIM, T_data> biCGstab2Kernel;
        BiCGstab3Kernel<DIM, T_data> biCGstab3Kernel;
        
        UpdateFieldWith1FieldKernelV2<DIM, T_data> updateFieldWith1FieldKernel;
        UpdateFieldWith2FieldsKernelV2<DIM, T_data> updateFieldWith2FieldsKernel;


        if(this->my_rank_==0)
        {
            std::cout << "Rank " << this->blockGrid_.getMyrank() << "\n extent" << this->alpakaHelper_.extent_  << 
                        "\n pitches" << " "<< pitchBuffAcc << " stride_j "<<stride_j_alpaka << " stride_k " << stride_k_alpaka << 
                        "\n workDivExtentFixed:\n\t" << workDivExtentFixed << 
                        "\n workDivExtentKernel1:\n\t" << workDivExtentKernel1 << 
                        "\n workDivExtentUpdateField1:\n\t" << workDivExtentUpdateField1 << 
                        "\n workDivExtentUpdateField2:\n\t" << workDivExtentUpdateField2 << std::endl;
        }

        rho0=1;
        rho1=rho0;
        preconditioner(MpkDev,pkDev);
        alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, biCGstab1Kernel, AMpkMdSpan, MpkMdSpan, r0MdSpan, ptrDotPorductDev1, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
        auto startSolver = std::chrono::high_resolution_clock::now();
        auto startCounter = std::chrono::high_resolution_clock::now();
        
        while(iter<maxIteration)
        {   
            startCounter = std::chrono::high_resolution_clock::now();
            preconditioner(MpkDev,pkDev);
            this->timeCounter.timeTotPreconditioner1 += std::chrono::high_resolution_clock::now() - startCounter;

            startCounter = std::chrono::high_resolution_clock::now();
            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(MpkDev));
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            this->timeCounter.timeTotCommunication1 += std::chrono::high_resolution_clock::now() - startCounter;

            startCounter = std::chrono::high_resolution_clock::now();
            this-> template resetNeumanBCsAlpaka<isMainLoop,false>(MpkDev);
            this->timeCounter.timeTotResetNeumanBCs += std::chrono::high_resolution_clock::now() - startCounter;
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
            
            // kernel1
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, biCGstab1Kernel, AMpkMdSpan, MpkMdSpan, r0MdSpan, ptrDotPorductDev1, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            this->timeCounter.timeTotKernel1 += std::chrono::high_resolution_clock::now() - startCounter;
            
            startCounter = std::chrono::high_resolution_clock::now();
            if constexpr(communicationON)
            {
                MPI_Allreduce(ptrDotPorductDev1, ptrTotSumDev2D, 1, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                memcpy(this->queueSolver_, totSumView, totSumDev2D, extent1D2);
                alphak=rho0/totSum[0];
            }
            else
            {
                memcpy(this->queueSolver_, dotPorductHost1, dotPorductDev1, extent1D);
                pSum[1] = ptrDotPorductHost1[0];
                alphak=rho0/pSum[1];
            }
            this->timeCounter.timeTotAllReduction1 += std::chrono::high_resolution_clock::now() - startCounter;
            

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

            // kernel2
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentFixed, updateFieldWith1FieldKernel, rkMdSpan, AMpkMdSpan,1, -alphak,
                                this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_);
            this->timeCounter.timeTotKernel2 += std::chrono::high_resolution_clock::now() - startCounter;
        
            startCounter = std::chrono::high_resolution_clock::now();
            preconditioner(zkDev,rkDev);
            this->timeCounter.timeTotPreconditioner2 += std::chrono::high_resolution_clock::now() - startCounter;
            startCounter = std::chrono::high_resolution_clock::now();
            if constexpr (communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(zkDev));
                this->communicatorMPI_.waitAllandCheckRcv();
            }
            this->timeCounter.timeTotCommunication2 += std::chrono::high_resolution_clock::now() - startCounter;
            
            startCounter = std::chrono::high_resolution_clock::now();
            this-> template resetNeumanBCsAlpaka<isMainLoop,false>(zkDev);
            this->timeCounter.timeTotResetNeumanBCs += std::chrono::high_resolution_clock::now() - startCounter;
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

            // kernel3
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2, biCGstab2Kernel, AzkMdSpan, zkMdSpan, rkMdSpan, ptrDotPorductDev2D, 
                                    this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);    
            this->timeCounter.timeTotKernel3 += std::chrono::high_resolution_clock::now() - startCounter;                    
            
            startCounter = std::chrono::high_resolution_clock::now();
            if  constexpr (communicationON)
            {
                MPI_Allreduce(ptrDotPorductDev2D, ptrTotSumDev2D, 2, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                memcpy(this->queueSolver_, totSumView, totSumDev2D, extent1D2);
                omegak=totSum[0]/totSum[1];  
            }
            else
            {
                memcpy(this->queueSolver_, dotPorductHost2D, dotPorductDev2D, extent1D2);
                pSum[0] = *ptrDotPorductHost2D;
                pSum[1] = *(ptrDotPorductHost2D+1);
                omegak=pSum[0]/pSum[1];
            }
            this->timeCounter.timeTotAllReduction2 += std::chrono::high_resolution_clock::now() - startCounter;

            /*
            for(i=0; i< this->ntotlocal_guards_; i++)
            {
                fieldX[i] = fieldX[i] + alphak*Mpk[i] + omegak*zk[i];
            }
            */

            // kernel4
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentFixed, updateFieldWith2FieldsKernel, fieldXMdSpan, MpkMdSpan, zkMdSpan, 1, alphak, omegak,
                                this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.offsetSolver_);
            this->timeCounter.timeTotKernel4 += std::chrono::high_resolution_clock::now() - startCounter;
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
            // kernel 5
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel3, biCGstab3Kernel, rkMdSpan, AzkMdSpan, r0MdSpan, ptrDotPorductDev2D, omegak,
                                this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            this->timeCounter.timeTotKernel5 += std::chrono::high_resolution_clock::now() - startCounter;

            
            startCounter = std::chrono::high_resolution_clock::now();
            if constexpr (communicationON)
            {
                MPI_Allreduce(ptrDotPorductDev2D,ptrTotSumDev2D, 2, getMPIType<T_data>(), MPI_SUM, MPI_COMM_WORLD);
                memcpy(this->queueSolver_, totSumView, totSumDev2D, extent1D2);
                rho1=totSum[0];
                this->errorFromIteration_ = std::sqrt(totSum[1]);
            }
            else
            {
                memcpy(this->queueSolver_, dotPorductHost2D, dotPorductDev2D, extent1D2);
                pSum[0] = *ptrDotPorductHost2D;
                pSum[1] = *(ptrDotPorductHost2D+1);
                this->errorFromIteration_ = std::sqrt(pSum[1]);
                rho1=pSum[0];
            }
             this->timeCounter.timeTotAllReduction2 += std::chrono::high_resolution_clock::now() - startCounter;
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

            // kernel 6
            startCounter = std::chrono::high_resolution_clock::now();
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentFixed, updateFieldWith2FieldsKernel, pkMdSpan, rkMdSpan, AMpkMdSpan, betak, 1, -betak*omegak, 
                                this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_);
             this->timeCounter.timeTotKernel6 += std::chrono::high_resolution_clock::now() - startCounter;    

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

        this->numIterationFinal_ = iter;
        
        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(fieldXDev));
            this->communicatorMPI_.waitAllandCheckRcv();
        }

        this-> template resetNeumanBCsAlpaka<isMainLoop,true>(fieldXDev);
        auto endSolver = std::chrono::high_resolution_clock::now();
        this->durationSolver_ = endSolver - startSolver;

        if constexpr (isMainLoop)
        {
            this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(fieldXDev,fieldBDev,rkDev);
        }


        this-> template resetNormFieldsAlpaka<communicationON>(fieldXDev,fieldBDev,this->normFieldB_);
        this->normFieldB_ = 1;

        memcpy(this->queueSolver_, fieldXHostView, fieldXDev, this->alpakaHelper_.extent_);
        alpaka::wait(this->queueSolver_);
    }

    private:
    T_Preconditioner preconditioner;
    const T_data toll_;

    alpaka::Buf<T_Dev, T_data, Dim, Idx> pkDev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> rkDev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> r0Dev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> MpkDev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> AMpkDev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> zkDev;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> AzkDev;

    alpaka::Vec<alpaka::DimInt<3>, Idx> pitchBuffAcc;

    Idx stride_j_alpaka;
    Idx stride_k_alpaka;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentFixed;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel3;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentUpdateField1;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentUpdateField2;
};