#!/usr/bin/env bash

# Install OpenCL on Ubuntu Xenial (16.04) on Intel Device

set -ev

apt-get -yq update
apt install -yq software-properties-common
add-apt-repository -y ppa:ubuntu-toolchain-r/test
apt-get -yq update

# Utilities & Clang & OpenCL ICD
apt-get install -yq --allow-downgrades --allow-remove-essential --allow-change-held-packages \
    git wget apt-utils cmake unzip libboost-all-dev clinfo cpio \
    clang-6.0 libomp-dev \
    ocl-icd-opencl-dev ocl-icd-dev opencl-headers

# https://software.intel.com/en-us/articles/opencl-drivers#latest_CPU_runtime
ICD_PACKAGE_URL=http://registrationcenter-download.intel.com/akdlm/irc_nas/12556/opencl_runtime_16.1.2_x64_rh_6.4.0.37.tgz
ICD_PACKAGE_NAME=opencl_runtime_16.1.2_x64_rh_6.4.0.37

wget -q ${ICD_PACKAGE_URL} -O /tmp/opencl_runtime.tgz && tar -xzf /tmp/opencl_runtime.tgz -C /tmp

sed 's/decline/accept/g' -i /tmp/${ICD_PACKAGE_NAME}/silent.cfg
/tmp/${ICD_PACKAGE_NAME}/install.sh -s /tmp/${ICD_PACKAGE_NAME}/silent.cfg

clinfo
