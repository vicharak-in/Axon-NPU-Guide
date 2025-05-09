#!/usr/bin/bash

BUILD_TYPE=Release
TARGET_PLATFORM=linux

GCC_COMPILER_PATH=/usr/bin/aarch64-linux-gnu

#GCC_COMPILER_PATH=/usr/bin/
C_COMPILER=${GCC_COMPILER_PATH}-gcc
CXX_COMPILER=${GCC_COMPILER_PATH}-g++
STRIP_COMPILER=${GCC_COMPILER_PATH}-strip

TARGET_ARCH=aarch64
TARGET_PLATFORM=linux

BUILD_DIR=$(pwd)/build
echo $BUILD_DIR

mkdir -p ${BUILD_DIR}

cd ${BUILD_DIR}

cmake .. \
    -DCMAKE_SYSTEM_PROCESSOR=${TARGET_ARCH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_C_COMPILER=${C_COMPILER} \
    -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DLIBRKNNRT=/usr/lib/librknnrt.so

make -j8