# FDA VICTRE_PIPELINE, unmodified, in the environment of its own CI
# (.github/workflows/test.yml: Ubuntu 22.04, VTK 7, boost, BLAS/LAPACK, TetGen).
# The C++ tools are built in place, where Victre.Pipeline expects them
# (Victre/generation/build/breastPhantomMain, Victre/compression/build/breastCompressMain,
# Victre/breastMass/build/breastMass), and the Python package is importable:
#
#   docker build -t breastsim/victre src/docker/victre
#   docker run --rm -u $(id -u):$(id -g) -v $PWD/work:/work breastsim/victre \
#       python3 /work/generate.py          # uses: from Victre import Pipeline
#
# BreastSIM only calls the anatomy steps (generate_phantom, insert_lesions, optionally
# compress_phantom/crop); the X-ray steps (project, reconstruct; CUDA) are not used.
FROM ubuntu:22.04

ARG VICTRE_PIPELINE_COMMIT=d3caa79

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git ca-certificates cmake zlib1g-dev gzip \
        libblas-dev liblapack-dev libopenmpi-dev libboost-dev libboost-program-options-dev \
        libproj-dev libtet1.5-dev fftw3 vtk7 libvtk7-dev libomp5 libomp-dev qtbase5-dev \
        python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/DIDSR/VICTRE_PIPELINE /opt/VICTRE_PIPELINE \
    && git -C /opt/VICTRE_PIPELINE checkout ${VICTRE_PIPELINE_COMMIT} \
    && python3 -m pip install --no-cache-dir -r /opt/VICTRE_PIPELINE/requirements.txt

# same build steps as the upstream CI
RUN cd /opt/VICTRE_PIPELINE \
    && mkdir ./Victre/generation/build && (cd ./Victre/generation/build && cmake .. && make -j"$(nproc)") \
    && mkdir ./Victre/compression/build && (cd ./Victre/compression/build && cmake .. && make -j"$(nproc)") \
    && mkdir ./Victre/breastMass/build && (cd ./Victre/breastMass/build && cmake .. && make -j"$(nproc)") \
    && chmod -R a+rX /opt/VICTRE_PIPELINE

ENV PYTHONPATH=/opt/VICTRE_PIPELINE
WORKDIR /opt/VICTRE_PIPELINE
