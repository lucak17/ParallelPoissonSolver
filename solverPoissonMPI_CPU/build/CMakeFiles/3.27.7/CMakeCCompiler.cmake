set(CMAKE_C_COMPILER "/opt/cray/pe/craype/2.7.30/bin/cc")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "17.0.3")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "CrayPrgEnv")
set(CMAKE_C_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_C_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_C_COMPILE_FEATURES "c_std_90;c_function_prototypes;c_std_99;c_restrict;c_variadic_macros;c_std_11;c_static_assert;c_std_17;c_std_23")
set(CMAKE_C90_COMPILE_FEATURES "c_std_90;c_function_prototypes")
set(CMAKE_C99_COMPILE_FEATURES "c_std_99;c_restrict;c_variadic_macros")
set(CMAKE_C11_COMPILE_FEATURES "c_std_11;c_static_assert")
set(CMAKE_C17_COMPILE_FEATURES "c_std_17")
set(CMAKE_C23_COMPILE_FEATURES "c_std_23")

set(CMAKE_C_PLATFORM_ID "Linux")
set(CMAKE_C_SIMULATE_ID "")
set(CMAKE_C_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_C_SIMULATE_VERSION "")




set(CMAKE_AR "/opt/cray/pe/cce/17.0.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ar")
set(CMAKE_C_COMPILER_AR "CMAKE_C_COMPILER_AR-NOTFOUND")
set(CMAKE_RANLIB "/opt/cray/pe/cce/17.0.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ranlib")
set(CMAKE_C_COMPILER_RANLIB "CMAKE_C_COMPILER_RANLIB-NOTFOUND")
set(CMAKE_LINKER "/opt/cray/pe/cce/17.0.0/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld")
set(CMAKE_MT "")
set(CMAKE_TAPI "CMAKE_TAPI-NOTFOUND")
set(CMAKE_COMPILER_IS_GNUCC )
set(CMAKE_C_COMPILER_LOADED 1)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ABI_COMPILED TRUE)

set(CMAKE_C_COMPILER_ENV_VAR "CC")

set(CMAKE_C_COMPILER_ID_RUN 1)
set(CMAKE_C_SOURCE_FILE_EXTENSIONS c;m)
set(CMAKE_C_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_C_LINKER_PREFERENCE 10)
set(CMAKE_C_LINKER_DEPFILE_SUPPORTED TRUE)

# Save compiler ABI information.
set(CMAKE_C_SIZEOF_DATA_PTR "8")
set(CMAKE_C_COMPILER_ABI "ELF")
set(CMAKE_C_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_C_LIBRARY_ARCHITECTURE "")

if(CMAKE_C_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_C_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_C_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_C_COMPILER_ABI}")
endif()

if(CMAKE_C_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_C_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_C_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/include;/opt/cray/pe/libsci/23.12.5/CRAY/17.0/x86_64/include;/opt/rocm-5.7.0/include;/opt/rocm-5.7.0/rocprofiler/include;/opt/rocm-5.7.0/roctracer/include;/opt/rocm-5.7.0/hip/include;/opt/cray/xpmem/2.8.2-1.0_3.9__g84a27a5.shasta/include;/opt/cray/pe/dsmml/0.2.2/dsmml/include;/opt/cray/pe/cce/17.0.0/cce-clang/x86_64/lib/clang/17/include;/opt/cray/pe/cce/17.0.0/cce/x86_64/include/craylibs;/usr/local/include;/usr/x86_64-suse-linux/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "amdhip64;sci_cray_mpi;sci_cray;dl;mpi_cray;dsmml;xpmem;stdc++;pgas-shmem;quadmath;modules;fi;craymath;f;u;csup;pthread;atomic;m;unwind;c;unwind")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/mpich/8.1.28/ofi/cray/17.0/lib;/opt/cray/pe/libsci/23.12.5/CRAY/17.0/x86_64/lib;/opt/rocm-5.7.0/lib64;/opt/rocm-5.7.0/lib;/opt/rocm-5.7.0/rocprofiler/lib;/opt/rocm-5.7.0/rocprofiler/tool;/opt/rocm-5.7.0/roctracer/lib;/opt/rocm-5.7.0/roctracer/tool;/opt/rocm-5.7.0/hip/lib;/opt/cray/xpmem/2.8.2-1.0_3.9__g84a27a5.shasta/lib64;/opt/cray/pe/dsmml/0.2.2/dsmml/lib;/opt/cray/pe/cce/17.0.0/cce/x86_64/lib;/usr/lib64/gcc/x86_64-suse-linux/12;/usr/lib64;/lib64;/usr/x86_64-suse-linux/lib;/lib;/usr/lib;/opt/cray/pe/cce/17.0.0/cce-clang/x86_64/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
