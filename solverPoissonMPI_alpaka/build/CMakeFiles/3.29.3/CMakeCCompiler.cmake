set(CMAKE_C_COMPILER "/opt/rocm-6.0.3/llvm/bin/clang")
set(CMAKE_C_COMPILER_ARG1 "")
set(CMAKE_C_COMPILER_ID "Clang")
set(CMAKE_C_COMPILER_VERSION "17.0.0")
set(CMAKE_C_COMPILER_VERSION_INTERNAL "")
set(CMAKE_C_COMPILER_WRAPPER "")
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




set(CMAKE_AR "/opt/rocm-6.0.3/llvm/bin/llvm-ar")
set(CMAKE_C_COMPILER_AR "/opt/rocm-6.0.3/lib/llvm/bin/llvm-ar")
set(CMAKE_RANLIB "/opt/rocm-6.0.3/llvm/bin/llvm-ranlib")
set(CMAKE_C_COMPILER_RANLIB "/opt/rocm-6.0.3/lib/llvm/bin/llvm-ranlib")
set(CMAKE_LINKER "/opt/rocm-6.0.3/llvm/bin/ld.lld")
set(CMAKE_LINKER_LINK "")
set(CMAKE_LINKER_LLD "")
set(CMAKE_C_COMPILER_LINKER "/opt/rocm-6.0.3/llvm/bin/ld.lld")
set(CMAKE_C_COMPILER_LINKER_ID "LLD")
set(CMAKE_C_COMPILER_LINKER_VERSION 17.0.0)
set(CMAKE_C_COMPILER_LINKER_FRONTEND_VARIANT GNU)
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
set(CMAKE_C_LINKER_DEPFILE_SUPPORTED FALSE)

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





set(CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES "/appl/lumi/SW/LUMI-24.03/common/EB/buildtools/24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/ICU/74.1-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/zstd/1.5.5-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/lz4/1.9.4-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/XZ/5.4.4-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/zlib/1.3.1-cpeCray-24.03/include;/appl/lumi/SW/LUMI-24.03/G/EB/bzip2/1.0.8-cpeCray-24.03/include;/opt/rocm-6.0.3/lib/llvm/lib/clang/17.0.0/include;/usr/local/include;/usr/x86_64-suse-linux/include;/usr/include")
set(CMAKE_C_IMPLICIT_LINK_LIBRARIES "mpi;mpi_gtl_hsa;gcc_s;c;gcc_s")
set(CMAKE_C_IMPLICIT_LINK_DIRECTORIES "/opt/cray/pe/mpich/8.1.29/ofi/crayclang/17.0/lib;/opt/cray/pe/mpich/8.1.29/gtl/lib;/usr/lib64/gcc/x86_64-suse-linux/13;/usr/lib64;/lib64;/usr/x86_64-suse-linux/lib;/lib;/usr/lib;/appl/lumi/SW/LUMI-24.03/common/EB/buildtools/24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/Boost/1.83.0-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/ICU/74.1-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/zstd/1.5.5-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/lz4/1.9.4-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/XZ/5.4.4-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/zlib/1.3.1-cpeCray-24.03/lib;/appl/lumi/SW/LUMI-24.03/G/EB/bzip2/1.0.8-cpeCray-24.03/lib")
set(CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
