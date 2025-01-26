#include <array>
#include <iostream>
#include <cstring>
#include <cmath> 

#pragma once
#include <alpaka/alpaka.hpp>
#include "iterativeSolverBaseAlpaka.hpp"
#include "blockGrid.hpp"
#include "solverSetup.hpp"
#include "communicationMPI.hpp"
#include "alpakaHelper.hpp"
#include "kernelsAlpakaChebyshev.hpp"
//#include "operationGrid.hpp"

template <int DIM, typename T_data, typename T_data_chebyshev, bool global, int tolerance, int maxIteration, bool isMainLoop, bool communicationON, typename T_Preconditioner>
class ChebyshevIterationAlpaka : public IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>{
    
    public:

    ChebyshevIterationAlpaka(const BlockGrid<DIM,T_data>& blockGrid, const ExactSolutionAndBCs<DIM,T_data>& exactSolutionAndBCs, 
                        CommunicatorMPI<DIM,T_data>& communicatorMPI, const AlpakaHelper<DIM,T_data>& alpakaHelper):
        IterativeSolverBaseAlpaka<DIM,T_data,maxIteration>(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        preconditioner(blockGrid,exactSolutionAndBCs,communicatorMPI,alpakaHelper),
        theta_(0),
        delta_(0),
        sigma_(0),
        thetaGlobal_( (this->eigenValuesGlobal_[0]*rescaleEigMin + this->eigenValuesGlobal_[1]*rescaleEigMax) * 0.5 * (1.0 + epsilon) ),
        deltaGlobal_( (this->eigenValuesGlobal_[1]*rescaleEigMax - this->eigenValuesGlobal_[0]*rescaleEigMin) * 0.5 ),
        thetaLocal_( (this->eigenValuesLocal_[0] + this->eigenValuesLocal_[1]) * 0.5),
        deltaLocal_( (this->eigenValuesLocal_[1] - this->eigenValuesLocal_[0]) * 0.5 ),
        sigmaGlobal_( thetaGlobal_/deltaGlobal_ ),
        sigmaLocal_( thetaLocal_/deltaLocal_ ),
        toll_(static_cast<T_data>(tolerance) * tollScalingFactor),
        bufTmp(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufBTmp(alpaka::allocBuf<T_data_chebyshev, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufY(alpaka::allocBuf<T_data_chebyshev, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufW(alpaka::allocBuf<T_data_chebyshev, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        bufZ(alpaka::allocBuf<T_data_chebyshev, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        rkDev(alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_)),
        workDivExtentCast(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentHelper_, this->alpakaHelper_.devAcc_, castPrecisionFieldKernel, 
                                                            alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufY))),
        workDivExtentKernel1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev1Kernel, 
                                                            alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_) ),
        workDivExtentKernel1Shared(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev1KernelShared, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_) ),
        workDivExtentKernel2Shared(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentNoHaloHelper_, this->alpakaHelper_.devAcc_, chebyshev2KernelShared, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_)),
        workDivExtentKernel1Solver(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev1KernelSharedMemSolver, 
                                                            alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufY), alpaka::experimental::getMdSpan(this->bufZ), delta_, theta_, theta_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentKernel2Fixed(alpaka::WorkDivMembers<Dim,Idx>(this->alpakaHelper_.numBlocksGrid_,blockExtentFixed,this->alpakaHelper_.elemPerThread_)),
        workDivExtentKernel2Solver(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev2KernelSharedMemSolver, 
                                                           alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentKernel2SharedMemeLoop1D(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_ , this->alpakaHelper_.devAcc_, chebyshev2KernelSharedMemLoop1D, 
                                                           alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufY),alpaka::experimental::getMdSpan(this->bufW), alpaka::experimental::getMdSpan(this->bufZ), delta_, sigma_, sigma_, sigma_,
                                                            this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_)),
        workDivExtentAssign1(alpaka::getValidWorkDiv(this->alpakaHelper_.cfgExtentSolverHelper_, this->alpakaHelper_.devAcc_, assignFieldWith1FieldKernel, alpaka::experimental::getMdSpan(this->bufTmp), alpaka::experimental::getMdSpan(this->bufY), static_cast<T_data_chebyshev>(1.0)/theta_, 
                            this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_))
    {
        if constexpr(global)
        {
            theta_ = thetaGlobal_;
            delta_ = deltaGlobal_;
            sigma_ = sigmaGlobal_;
        }
        else
        {
            theta_ = thetaLocal_;
            delta_ = deltaLocal_;
            sigma_ = sigmaLocal_;
        }
        constexpr std::uint8_t value0{0}; 
        alpaka::memset(this->queueSolverNonBlocking1_, bufTmp, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufBTmp, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufY, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufW, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, bufZ, value0);
        alpaka::memset(this->queueSolverNonBlocking1_, rkDev, value0);


        if(this->my_rank_==0)
        {
            std::cout << "Chebyshev preconditioner" <<
                        "\n workDivExtentKernel1Solver:\t" << workDivExtentKernel1Solver <<
                        "\n workDivExtentKernel2Solver:\t" << workDivExtentKernel2Solver <<
                        "\n workDivExtentKernel2SharedMemeLoop1D:\t" << workDivExtentKernel2SharedMemeLoop1D <<
                        "\n workDivExtentKernel2Shared:\t" << workDivExtentKernel2Shared <<
                        "\n workDivExtentKernel2Fixed:\t" << workDivExtentKernel2Fixed  << std::endl;
            std::cout<< "Min eigenvalue "<< this->eigenValuesGlobal_[0]*rescaleEigMin << "\nMax eigenvalue "<< this->eigenValuesGlobal_[1]*rescaleEigMax << std::endl;
        }

        alpaka::wait(this->queueSolverNonBlocking1_);

        // To do --> fix the datatype in case of low precision preconditioner with communicationON=true
        if constexpr (isMainLoop && communicationON)
        {   
            auto pitchBuffAcc = alpaka::getPitchesInBytes(bufY);
            for(int i=0; i<3; i++)
            {
                pitchBuffAcc[i] /= sizeof(T_data);
            }
            Idx stride_j_alpaka = pitchBuffAcc[1];
            Idx stride_k_alpaka = pitchBuffAcc[0];
            //int ntotBuffAcc = stride_k_alpaka * this->alpakaHelper_.extent_[0];
            this-> communicatorMPI_.setStridesAlpaka(stride_j_alpaka,stride_k_alpaka);
        }
    }

    ~ChebyshevIterationAlpaka()
    {
    }

    void operator()(alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufX, alpaka::Buf<T_Dev, T_data, Dim, Idx>& bufB)
    {   
        constexpr std::uint8_t value0{0};
        T_data_chebyshev rhoOld=1/sigma_;
        T_data_chebyshev rhoCurr=1/(2*sigma_ - rhoOld);

        // to do fix neuman BCs for mixed precision
        this-> template resetNeumanBCsAlpaka<isMainLoop,false>(bufB);
        if constexpr(communicationON)
        {
            this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufB));
            this->communicatorMPI_.waitAllandCheckRcv();
        }


        auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);
        auto bufBTmpMdSpan = alpaka::experimental::getMdSpan(this->bufBTmp);

        auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
        auto bufYMdSpan = alpaka::experimental::getMdSpan(this->bufY);
        auto bufWMdSpan = alpaka::experimental::getMdSpan(this->bufW);
        auto bufZMdSpan = alpaka::experimental::getMdSpan(this->bufZ);


        auto startSolver = std::chrono::high_resolution_clock::now();

        
        
        constexpr bool cast = true;
        if constexpr (!std::is_same_v<T_data, T_data_chebyshev> && cast )
        {
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentCast, castPrecisionFieldKernel, bufBMdSpan, bufBTmpMdSpan);
            
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, chebyshev1Kernel, bufBTmpMdSpan, bufYMdSpan, bufZMdSpan, 
                    delta_, theta_, rhoCurr, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            
            for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
            {   
                rhoOld=rhoCurr;
                rhoCurr=1/(2*sigma_ - rhoOld);
                
                // to do fix neuman BCs mixed precision
                this-> template resetNeumanBCsAlpakaCast<T_data_chebyshev,isMainLoop,false>(bufY);
                
                if constexpr(communicationON)
                {
                    this->communicatorMPI_.template operator()<T_data_chebyshev,true>(getPtrNative(bufY));
                    this->communicatorMPI_.waitAllandCheckRcv();
                }
                
                if constexpr (jumpCheb == 1)
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2SharedMemeLoop1D, chebyshev2KernelSharedMemLoop1D, bufBTmpMdSpan, bufYMdSpan,bufWMdSpan, 
                                                bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                                this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_);
                }
                else if constexpr (jumpCheb == 2)
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Shared, chebyshev2KernelShared, bufBTmpMdSpan, bufYMdSpan,bufWMdSpan, 
                                            bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                            this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
                }
                else
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Solver, chebyshev2Kernel, bufBTmpMdSpan, bufYMdSpan,bufWMdSpan, 
                                    bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
                }
                std::swap(this->bufZ,this->bufY);
                std::swap(bufZMdSpan,bufYMdSpan);
                std::swap(this->bufW,this->bufY);
                std::swap(bufWMdSpan,bufYMdSpan);
                
            }
        }
        else
        {
            // apply s0 and s1
            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        fieldZ[indx] = fieldB[indx]/theta_;
                        fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    }
                }
            }
            */
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, chebyshev1Kernel, bufBMdSpan, bufYMdSpan, bufZMdSpan, 
                        delta_, theta_, rhoCurr, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            
            // apply si
            /*
            for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
            {   
                rhoOld=rhoCurr;
                rhoCurr=1/(2*sigma_ - rhoOld);
                if constexpr (communicationON)
                {
                    this->communicatorMPI_(fieldY);
                    this->communicatorMPI_.waitAllandCheckRcv();
                }
                this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
                for(k=kmin; k<kmax; k++)
                {
                    for(j=jmin; j<jmax; j++)
                    {
                        for(i=imin; i<imax; i++)
                        {
                            indx=i + this->stride_j_*j + this->stride_k_*k;
                            fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                        }
                    }
                }
                std::swap(fieldZ,fieldY);
                std::swap(fieldW,fieldY);
            }
            */
            
            for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
            {   
                rhoOld=rhoCurr;
                rhoCurr=1/(2*sigma_ - rhoOld);
                
                // to do fix neuman BCs mixed precision
                this-> template resetNeumanBCsAlpakaCast<T_data_chebyshev,isMainLoop,false>(bufY);
                
                if constexpr(communicationON)
                {
                    this->communicatorMPI_.template operator()<T_data_chebyshev,true>(getPtrNative(bufY));
                    this->communicatorMPI_.waitAllandCheckRcv();
                }
                
                if constexpr (jumpCheb == 1)
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2SharedMemeLoop1D, chebyshev2KernelSharedMemLoop1D, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                                bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                                this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_, this->alpakaHelper_.offsetSolver_);
                }
                else if constexpr (jumpCheb == 2)
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Shared, chebyshev2KernelShared, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                            bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, 
                                            this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
                }
                else
                {
                    alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Solver, chebyshev2Kernel, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                    bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
                }
                
                std::swap(this->bufZ,this->bufY);
                std::swap(bufZMdSpan,bufYMdSpan);
                std::swap(this->bufW,this->bufY);
                std::swap(bufWMdSpan,bufYMdSpan);
                
            }
            
        }
        
        
        

        /*
        for(k=kmin; k<kmax; k++)
        {
            for(j=jmin; j<jmax; j++)
            {
                for(i=imin; i<imax; i++)
                {
                    indx=i + this->stride_j_*j + this->stride_k_*k;
                    fieldX[indx] = (-1)*fieldW[indx];
                }
            }
        }
        */

        alpaka::exec<Acc>(this->queueSolver_, workDivExtentAssign1, assignFieldWith1FieldKernel, bufXMdSpan, bufWMdSpan, static_cast<T_data>(-1.0),
                         this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_);

        auto endSolver = std::chrono::high_resolution_clock::now();

        /*
        if constexpr (isMainLoop)
        {
            std::swap(fieldBtmp,fieldB);
            this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,fieldBtmp,operatorA);
            this->errorFromIteration_ = this->errorComputeOperator_;
            this->numIterationFinal_ = maxIteration;
            this->durationSolver_ = endSolver - startSolver;
        }
        */
    }

    void operator()(T_data fieldX[], T_data fieldB[])
    {   
        if constexpr(std::is_same<T_data, T_data_chebyshev>::value)
        {
            constexpr std::uint8_t value0{0};       
            T_data_chebyshev rhoOld=1/sigma_;
            T_data_chebyshev rhoCurr=1/(2*sigma_ - rhoOld);

            auto fieldXHostView = createView(this->alpakaHelper_.devHost_, fieldX, this->alpakaHelper_.extent_);
            auto fieldBHostView = createView(this->alpakaHelper_.devHost_, fieldB, this->alpakaHelper_.extent_);

            auto bufX = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);
            auto bufB = alpaka::allocBuf<T_data, Idx>(this->alpakaHelper_.devAcc_, this->alpakaHelper_.extent_);

            auto bufXMdSpan = alpaka::experimental::getMdSpan(bufX);
            auto bufBMdSpan = alpaka::experimental::getMdSpan(bufB);
            auto bufBTmpMdSpan = alpaka::experimental::getMdSpan(this->bufBTmp);
            auto bufYMdSpan = alpaka::experimental::getMdSpan(this->bufY);
            auto bufWMdSpan = alpaka::experimental::getMdSpan(this->bufW);
            auto bufZMdSpan = alpaka::experimental::getMdSpan(this->bufZ);

            if constexpr (isMainLoop)
            {
                this->setProblemAlpaka(bufX, bufB);
            }

            memcpy(this->queueSolver_, this->bufBTmp, bufB, this->alpakaHelper_.extent_);
            alpaka::wait(this->queueSolver_);
            
            /*
            if (isMainLoop && communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufX));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            if constexpr(isMainLoop)
            {
                this-> template resetNeumanBCsAlpaka<isMainLoop,true>(bufX);
            }
            else 
            {
                alpaka::memset(this->queueSolver_, bufX, value0);
            }

            this-> template normalizeProblemToFieldBNormAlpaka<isMainLoop,communicationON>(bufX,bufB);
            

            this-> template normalizeProblemToFieldBNormAlpaka<isMainLoop,communicationON>(bufX,this->bufBTmp);
            this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(bufX,this->bufBTmp,this->rkDev);
            std::cout<< "err: " << this->errorComputeOperator_<<std::endl;

            */

            memcpy(this->queueSolver_, this->bufBTmp, bufB, this->alpakaHelper_.extent_);
            alpaka::wait(this->queueSolver_);
            
            this -> adjustFieldBForDirichletNeumanBCsAlpaka(bufX, bufB);
            
            //this-> template resetNormFieldsAlpaka<communicationON>(this->bufZ,bufB,1/this->normFieldB_);

            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufB));
                this->communicatorMPI_.waitAllandCheckRcv();

                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(this->bufBTmp));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            this-> template resetNeumanBCsAlpaka<isMainLoop,false>(bufB);
            
            // r0= b - A(x0)
            this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(bufX,this->bufBTmp,this->rkDev);

            if constexpr(isMainLoop)
            {
                if(this->my_rank_==0)
                {    std::cout<<"Debug in chebyshev Alpaka START " << " main loop "<< isMainLoop << " globalLocation " << this->globalLocation_[0] << " "<< this->globalLocation_[1] <<" "<< this->globalLocation_[2] <<
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
                if(this->my_rank_==0)
                    std::cout<< "Exit befor loop, error: " << this->errorComputeOperator_  <<std::endl;
                return;
            }

                    

            // to do fix neuman BCs for mixed precision
            
            /*
            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufB));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            this-> template resetNeumanBCsAlpaka<isMainLoop,false>(bufB);
            */
            
            auto startSolver = std::chrono::high_resolution_clock::now();
            
            // apply s0 and s1
            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        fieldZ[indx] = fieldB[indx]/theta_;
                        fieldY[indx] = 2*rhoCurr/delta_ * (2*fieldB[indx] + operatorA(i,j,k,fieldB)/theta_ );
                    }
                }
            }
            */
            

            
            alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel1, chebyshev1Kernel, bufBMdSpan, bufYMdSpan, bufZMdSpan, 
                        delta_, theta_, rhoCurr, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.haloSize_);
            



            // apply si
            /*
            for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
            {   
                rhoOld=rhoCurr;
                rhoCurr=1/(2*sigma_ - rhoOld);
                if constexpr (communicationON)
                {
                    this->communicatorMPI_(fieldY);
                    this->communicatorMPI_.waitAllandCheckRcv();
                }
                this-> template resetNeumanBCs<isMainLoop,false>(fieldY);
                for(k=kmin; k<kmax; k++)
                {
                    for(j=jmin; j<jmax; j++)
                    {
                        for(i=imin; i<imax; i++)
                        {
                            indx=i + this->stride_j_*j + this->stride_k_*k;
                            fieldW[indx] = rhoCurr * ( 2*sigma_*fieldY[indx] +  2/delta_ * ( fieldB[indx] + operatorA(i,j,k,fieldY)  )  - rhoOld*fieldZ[indx] );
                        }
                    }
                }
                std::swap(fieldZ,fieldY);
                std::swap(fieldW,fieldY);
            }
            */
            
            for(int indxCheb=2; indxCheb<=maxIteration; indxCheb++)
            {   
                /*
                if(this->my_rank_==0)
                    std::cout<< "Chebyshev iteration i: " << indxCheb <<std::endl;
                */

                rhoOld=rhoCurr;
                rhoCurr=1/(2*sigma_ - rhoOld);
                
                
                if constexpr(communicationON)
                {
                    this->communicatorMPI_.template operator()<T_data_chebyshev,true>(getPtrNative(bufY));
                    this->communicatorMPI_.waitAllandCheckRcv();
                }

                // to do fix neuman BCs mixed precision
                this-> template resetNeumanBCsAlpakaCast<T_data_chebyshev,isMainLoop,false>(bufY);
                
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentKernel2Solver, chebyshev2Kernel, bufBMdSpan, bufYMdSpan,bufWMdSpan, 
                                    bufZMdSpan, delta_, sigma_, rhoCurr, rhoOld, this->alpakaHelper_.indexLimitsSolverAlpaka_, this->alpakaHelper_.ds_, this->alpakaHelper_.offsetSolver_);
                
                std::swap(this->bufZ,this->bufY);
                std::swap(bufZMdSpan,bufYMdSpan);
                std::swap(this->bufW,this->bufY);
                std::swap(bufWMdSpan,bufYMdSpan);
                /*
                alpaka::exec<Acc>(this->queueSolver_, workDivExtentAssign1, assignFieldWith1FieldKernel, bufXMdSpan, bufWMdSpan, static_cast<T_data_chebyshev>(-1.0),
                            this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_);

                if constexpr(communicationON)
                {
                    this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufX));
                    this->communicatorMPI_.waitAllandCheckRcv();
                }

                this-> template resetNeumanBCsAlpaka<isMainLoop,true>(bufX);
                this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(bufX,this->bufBTmp,this->rkDev);

                if(this->my_rank_==0)
                    std::cout<< "Chebyshev iter: "<< indxCheb << " rhoCurr: " << rhoCurr << " error: " << this->errorComputeOperator_  <<std::endl;
                */
            }

            /*
            for(k=kmin; k<kmax; k++)
            {
                for(j=jmin; j<jmax; j++)
                {
                    for(i=imin; i<imax; i++)
                    {
                        indx=i + this->stride_j_*j + this->stride_k_*k;
                        fieldX[indx] = (-1)*fieldW[indx];
                    }
                }
            }
            */

            alpaka::exec<Acc>(this->queueSolver_, workDivExtentAssign1, assignFieldWith1FieldKernel, bufXMdSpan, bufWMdSpan, static_cast<T_data_chebyshev>(-1.0),
                            this->alpakaHelper_.indexLimitsSolverAlpaka_,this->alpakaHelper_.offsetSolver_);

            if constexpr(communicationON)
            {
                this->communicatorMPI_.template operator()<T_data,true>(getPtrNative(bufX));
                this->communicatorMPI_.waitAllandCheckRcv();
            }

            this-> template resetNeumanBCsAlpaka<isMainLoop,true>(bufX);
            auto endSolver = std::chrono::high_resolution_clock::now();
            this->durationSolver_ = endSolver - startSolver;

            memcpy(this->queueSolverNonBlocking1_, bufB, this->bufBTmp, this->alpakaHelper_.extent_);
            alpaka::wait(this->queueSolverNonBlocking1_);

            if constexpr (isMainLoop)
            {
                this->errorComputeOperator_ = this-> template computeErrorOperatorAAlpaka<isMainLoop, communicationON>(bufX,this->bufBTmp,this->rkDev);
            }


            //this-> template resetNormFieldsAlpaka<communicationON>(bufX,bufB,this->normFieldB_);
            this->normFieldB_ = 1;

            memcpy(this->queueSolver_, fieldXHostView, bufX, this->alpakaHelper_.extent_);
            alpaka::wait(this->queueSolver_);
            if(this->my_rank_==0)
                    std::cout<< "End chebyshev error: " << this->errorComputeOperator_  <<std::endl;


            /*
            if constexpr (isMainLoop)
            {
                std::swap(fieldBtmp,fieldB);
                this->errorComputeOperator_ = this-> template computeErrorOperatorA<isMainLoop, communicationON>(fieldX,fieldB,fieldBtmp,operatorA);
                this->errorFromIteration_ = this->errorComputeOperator_;
                this->numIterationFinal_ = maxIteration;
                this->durationSolver_ = endSolver - startSolver;
            }
            */
        }
    }

    
    
    private:

    

    T_Preconditioner preconditioner;
    
    T_data_chebyshev theta_;
    T_data_chebyshev delta_;
    T_data_chebyshev sigma_;

    const T_data_chebyshev thetaGlobal_;
    const T_data_chebyshev deltaGlobal_;
    const T_data_chebyshev thetaLocal_;
    const T_data_chebyshev deltaLocal_;
    const T_data_chebyshev sigmaGlobal_;
    const T_data_chebyshev sigmaLocal_;

    T_data toll_;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> bufTmp;
    alpaka::Buf<T_Dev, T_data_chebyshev, Dim, Idx> bufBTmp;
    alpaka::Buf<T_Dev, T_data_chebyshev, Dim, Idx> bufY;
    alpaka::Buf<T_Dev, T_data_chebyshev, Dim, Idx> bufW;
    alpaka::Buf<T_Dev, T_data_chebyshev, Dim, Idx> bufZ;
    alpaka::Buf<T_Dev, T_data, Dim, Idx> rkDev;

    CastPrecisionFieldKernel<DIM, T_data, T_data_chebyshev> castPrecisionFieldKernel;
    Chebyshev1Kernel<DIM,T_data,T_data_chebyshev> chebyshev1Kernel;
    Chebyshev2Kernel<DIM,T_data,T_data_chebyshev> chebyshev2Kernel;
    Chebyshev1KernelSharedMem<DIM,T_data,T_data_chebyshev> chebyshev1KernelShared;
    Chebyshev2KernelSharedMem<DIM,T_data,T_data_chebyshev> chebyshev2KernelShared;
    Chebyshev1KernelSharedMemSolver<DIM, T_data, T_data_chebyshev> chebyshev1KernelSharedMemSolver;
    Chebyshev2KernelSharedMemSolver<DIM, T_data, T_data_chebyshev> chebyshev2KernelSharedMemSolver;
    Chebyshev2KernelSharedMemLoop1D<DIM, T_data, T_data_chebyshev> chebyshev2KernelSharedMemLoop1D;
    AssignFieldWith1FieldKernel<DIM, T_data, T_data_chebyshev> assignFieldWith1FieldKernel;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentCast;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1Shared;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Shared;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel1Solver;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Fixed;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2Solver;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentKernel2SharedMemeLoop1D;
    alpaka::WorkDivMembers<Dim, Idx> const workDivExtentAssign1;
};
