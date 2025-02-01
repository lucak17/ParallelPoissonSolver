# How to use the Poisson solver:
clone the repository:
```
git clone https://github.com/lucak17/ParallelPoissonSolver
```
go to `ParallelPoissonSolver/solverPoissonMPI_alpaka/` folder
```
cd ParallelPoissonSolver/solverPoissonMPI_alpaka
```
create `build` and `run` folders
```
mkdir build && mkdir run
```
set up the environment according to the specific machine and back-end you are using. Currently available .profile files support Dardel-G, LUMI-C, LUMI-G and MareNostrum5-G\
e.g to run on LUMI-G
```
source env/lumi/solverPoisson_lumiGPU.profile
```
go to `build` folder
```
cd build
```
run `cmake`
```
cmake ..
```
build the solver
```
make
```
go to `run` folder
```
cd ../run
```
modify the `run.slurm` script and submit it\
The problem parameters and the solver properties are set in `solverSetup.hpp` and `inputParam.hpp` files in the `include` folder. The solver must be compiled again any time these files are changed.
