# VICTRE tools (breast phantom generation, compression, mass generation) built from the
# unmodified sources of the FDA's maintained VICTRE_PIPELINE repository, in the same
# environment and with the same commands as its own CI (.github/workflows/test.yml:
# Ubuntu 22.04, VTK 7, boost, BLAS/LAPACK, TetGen). No patches are applied.
#
# The archived standalone repositories (DIDSR/breastPhantom, DIDSR/breastMass) do not build
# as published (breastPhantom's artery/vein code needs VTK 5; breastMass HEAD references an
# undeclared variable); VICTRE_PIPELINE ships the authors' maintained copies of both.
#
#   docker build -t breastsim/victre src/docker/victre
#   docker run --rm -u $(id -u):$(id -g) -v $PWD/work:/work breastsim/victre breastPhantom -c /work/phantom.cfg
FROM ubuntu:22.04

ARG VICTRE_PIPELINE_COMMIT=d3caa79

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git ca-certificates cmake zlib1g-dev gzip \
        libblas-dev liblapack-dev libopenmpi-dev libboost-dev libboost-program-options-dev \
        libproj-dev libtet1.5-dev fftw3 vtk7 libvtk7-dev libomp5 libomp-dev qtbase5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DIDSR/VICTRE_PIPELINE /src/VICTRE_PIPELINE \
    && git -C /src/VICTRE_PIPELINE checkout ${VICTRE_PIPELINE_COMMIT}

# same steps as the upstream CI
RUN cd /src/VICTRE_PIPELINE \
    && mkdir ./Victre/generation/build && (cd ./Victre/generation/build && cmake .. && make -j"$(nproc)") \
    && mkdir ./Victre/compression/build && (cd ./Victre/compression/build && cmake .. && make -j"$(nproc)") \
    && mkdir ./Victre/breastMass/build && (cd ./Victre/breastMass/build && cmake .. && make -j"$(nproc)") \
    && install -m 755 ./Victre/generation/build/breastPhantom /usr/local/bin/breastPhantom \
    && install -m 755 ./Victre/breastMass/build/breastMass /usr/local/bin/breastMass \
    && find ./Victre/compression/build -maxdepth 1 -type f -executable -exec install -m 755 {} /usr/local/bin/ \;

WORKDIR /work
