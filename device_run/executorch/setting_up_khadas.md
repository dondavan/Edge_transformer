# Use miniconda
> conda create -n python_3_10 python=3.10
> source ~/miniconda3/bin/activate

# Install executorch runtime
https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html

# From the root of the executorch repo:
./install_executorch.sh --clean
git submodule sync
git submodule update --init --recursive

During cmake, some depedency are missing:

tomli
> pip3 install tomli
zstd
> pip3 install zstd
torch
> pip3 install torch
glslc
> sudo apt install glslc
yaml
> pip3 install PyYAML

Vulkan needs Android NDK
after installation
> export ANDROID_NDK=/home/khadas/android-ndk-r27c
> export ANDROID_ABI=arm64-v8a

cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake  -DANDROID_ABI=arm64-v8a ..

On laptop:
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=cmake-android-out \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ANDROID_ABI \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 

On device:
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON 


maybe: -DCMAKE_POLICY_VERSION_MINIMUM=3.5 

on Mac
> export ANDROID_NDK_HOME="/opt/homebrew/share/android-ndk"
> export ANDROID_NDK=/opt/homebrew/share/android-ndk

Configure cmake build:
> cmake .. -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_VULKAN=ON -DEXECUTORCH_BUILD_XNNPACK=ON
# 


/home/khadas/torch/cmake-out/models

#include <ostream>
#include <cstdint>

# GPU execution runner
./executorch/backends/vulkan/vulkan_executor_runner --model_path models/bert-base-uncased_GPU.pte


[ 1.000000e+00, 7.300000e+01, 2.100000e+01, 4.100000e+01,  ..., 9.600000e+01, 7.800000e+01, 8.400000e+01, 1.160000e+02, ] 
[ 1.030000e+02, 4.400000e+01, 8.000000e+01, 8.000000e+00,  ..., 8.700000e+01, 1.090000e+02, 1.000000e+01, 3.000000e+01, ] 
[ 9.100000e+01, 1.160000e+02, 4.800000e+01, 1.140000e+02,  ..., 1.040000e+02, 9.100000e+01, 7.100000e+01, 5.900000e+01, ] 
[ 1.220000e+02, 6.600000e+01, 1.400000e+01, 1.080000e+02,  ..., 6.500000e+01, 7.300000e+01, 9.000000e+00, 1.050000e+02, ] 
[ 3.900000e+01, 4.300000e+01, 9.600000e+01, 9.500000e+01,  ..., 1.030000e+02, 1.200000e+02, 9.600000e+01, 9.100000e+01, ] 
 ..., 
[ 9.000000e+00, 8.000000e+00, 8.100000e+01, 1.800000e+01,  ..., 2.100000e+01, 8.000000e+01, 9.500000e+01, 7.700000e+01, ] 
[ 1.500000e+01, 7.700000e+01, 8.900000e+01, 3.200000e+01,  ..., 7.800000e+01, 8.700000e+01, 1.160000e+02, 4.200000e+01, ] 
[ 1.500000e+01, 4.100000e+01, 2.000000e+00, 9.300000e+01,  ..., 8.100000e+01, 1.130000e+02, 9.600000e+01, 1.040000e+02, ] 
[ 3.000000e+01, 6.300000e+01, 3.200000e+01, 6.200000e+01,  ..., 1.030000e+02, 7.600000e+01, 1.160000e+02, 1.110000e+02, ] 
[ 8.000000e+01, 1.000000e+01, 5.500000e+01, 6.600000e+01,  ..., 1.050000e+02, 1.130000e+02, 6.500000e+01, 3.400000e+01, ]



[ -2.166991e-02, 1.075543e-02, -1.329368e-02, 2.443095e-02,  ..., 6.728482e-03, -2.401135e-02, 3.051479e-02, -4.914004e-02, ] 
[ -1.724559e-02, -1.521711e-02, -1.148175e-02, -6.233187e-03,  ..., 9.794910e-05, 4.095570e-02, 1.073183e-01, 3.873204e-02, ] 
[ -4.450352e-02, -4.914004e-02, 4.528401e-03, -6.554513e-02,  ..., 2.645693e-02, -4.450352e-02, 8.377831e-02, -5.154968e-02, ] 
[ 2.080218e-02, 5.576696e-02, -3.043689e-02, -1.060735e-01,  ..., 8.880432e-02, 1.075543e-02, 6.164173e-02, 1.757746e-02, ] 
[ 5.304072e-02, 7.525747e-02, 6.728482e-03, -4.071184e-02,  ..., -1.724559e-02, -3.222110e-02, 6.728482e-03, -4.450352e-02, ] 
 ..., 
[ 6.164173e-02, -6.233187e-03, -3.393076e-02, 2.301858e-03,  ..., -1.329368e-02, -1.148175e-02, -4.071184e-02, -2.847897e-02, ] 
[ 4.790619e-02, -2.847897e-02, 6.481855e-02, -3.564632e-02,  ..., -2.401135e-02, 9.794910e-05, -4.914004e-02, -2.631478e-02, ] 
[ 4.790619e-02, 2.443095e-02, 8.824104e-03, 5.862553e-02,  ..., -3.393076e-02, 1.253237e-02, 6.728482e-03, 2.645693e-02, ] 
[ 3.873204e-02, -8.067922e-03, -3.564632e-02, -4.218373e-03,  ..., -1.724559e-02, 2.252608e-02, -4.914004e-02, -3.904550e-02, ] 
[ -1.148175e-02, 1.073183e-01, -1.939630e-02, 5.576696e-02,  ..., 1.757746e-02, 1.253237e-02, 8.880432e-02, 1.422836e-02, ]

[ 3.190377e-02, -1.484348e-02, -5.588732e-02, 1.179120e-02,  ..., 2.375533e-02, -7.572627e-03, -6.063936e-02, 9.042858e-03, ] 
[ 5.932441e-04, -2.365437e-02, -8.390682e-03, 0.000000e+00,  ..., 3.519252e-02, 2.069087e-02, -5.621926e-03, 0.000000e+00, ] 
