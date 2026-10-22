# VICTRE breastPhantom / breastMass built from unmodified upstream sources in the
# environment documented by the authors (breastPhantom docs/installation.rst:
# Ubuntu 16.04, VTK 6, boost, BLAS/LAPACK). No patches are applied.
#
#   docker build -t breastsim/victre src/docker/victre
#   docker run --rm -u $(id -u):$(id -g) -v $PWD/work:/work breastsim/victre breastPhantom -c /work/phantom.cfg
FROM ubuntu:16.04

ARG BREASTPHANTOM_COMMIT=54f24e1
ARG BREASTMASS_COMMIT=245e55d

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        vtk6 libvtk6-dev libblas-dev liblapack-dev \
        libboost-dev libboost-program-options-dev libproj-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DIDSR/breastPhantom /src/breastPhantom \
    && git -C /src/breastPhantom checkout ${BREASTPHANTOM_COMMIT} \
    && mkdir /build-phantom && cd /build-phantom && cmake /src/breastPhantom && make -j"$(nproc)" \
    && install -m 755 breastPhantom /usr/local/bin/breastPhantom

RUN git clone https://github.com/DIDSR/breastMass /src/breastMass \
    && git -C /src/breastMass checkout ${BREASTMASS_COMMIT} \
    && mkdir /build-mass && cd /build-mass && cmake /src/breastMass && make -j"$(nproc)" \
    && install -m 755 breastMass /usr/local/bin/breastMass

WORKDIR /work
