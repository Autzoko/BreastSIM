# FEBio (official source, MIT license, github.com/febiosoftware/FEBio) with Intel MKL / PARDISO, plus the TetGen
# command line tool, for the anteroposterior compression of BreastSIM phantoms.
#
#   docker build --build-arg JOBS=8 -t breastsim/febio src/docker/febio
#   docker run --rm -u $(id -u):$(id -g) -e OMP_NUM_THREADS=8 -v $PWD/work:/work breastsim/febio febio4 -i /work/model.feb
#
# MKL and the Intel OpenMP runtime (libiomp5, needed by FEBio's MKL link line mkl_intel_thread) come from Intel's
# oneAPI apt repository (Ubuntu 22.04 has libmkl-dev but no libiomp5), pinned to 2024.2: MKL 2025 removed the
# deprecated sparse routines (mkl_dcsrtrsv) FEBio's ILU preconditioners call. FEBio's CMake takes them through
# MKL_INC / MKL_LIB_DIR / MKL_OMP_LIB.
FROM ubuntu:22.04

ARG FEBIO_TAG=v4.13
ARG JOBS=8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates curl gnupg zlib1g-dev tetgen \
    && curl -fsSL https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
        | gpg --dearmor -o /usr/share/keyrings/oneapi.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi.gpg] https://apt.repos.intel.com/oneapi all main" \
        > /etc/apt/sources.list.d/oneapi.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        intel-oneapi-mkl-devel-2024.2 intel-oneapi-openmp-2024.2 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${FEBIO_TAG} https://github.com/febiosoftware/FEBio /opt/FEBio \
    && cd /opt/FEBio && git rev-parse HEAD > /opt/FEBio/COMMIT \
    && MKL=/opt/intel/oneapi/mkl/2024.2 \
    && OMP=$(find /opt/intel/oneapi -name libiomp5.so | head -1) \
    && mkdir cbuild && cd cbuild \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_MKL=ON -DMKL_INC=$MKL/include -DMKL_LIB_DIR=$MKL/lib \
         -DMKL_OMP_LIB=$OMP \
    && make -j${JOBS} \
    && echo "$(dirname $OMP)" > /etc/ld.so.conf.d/iomp5.conf && echo "$MKL/lib" > /etc/ld.so.conf.d/mkl.conf && ldconfig \
    && ln -s $(find /opt/FEBio/cbuild/bin -maxdepth 1 -type f -name 'febio4*' | head -1) /usr/local/bin/febio4 \
    && chmod -R a+rX /opt/FEBio

WORKDIR /work
