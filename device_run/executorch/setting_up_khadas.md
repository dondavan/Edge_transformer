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