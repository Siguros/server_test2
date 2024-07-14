#!/usr/bin/env bash
cd "$HOME" || exit
git clone --recursive https://github.com/Simple-Robotics/proxsuite.git
cd proxsuite || exit
conda install cmake eigen simde -y
mkdir build && cd build || exit
cmake .. -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=ON -DBUILD_TESTING=OFF -DBUILD_WITH_OPENMP_SUPPORT=ON
make install -j"$(nproc)"
