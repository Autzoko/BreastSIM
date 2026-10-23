# breastsim/victre + FEBio, the finite element solver VICTRE's compression step (breastCompress)
# calls through $FEBIO_PATH. VICTRE documents FEBio 2.x (binary download, account required) and
# says 3.x "might also work but has not been tested"; this image builds FEBio 3.8 from the official
# open-source repository (MIT) with Intel MKL from Ubuntu multiverse (Pardiso solver; without MKL
# FEBio falls back to the serial Skyline solver, ~6 h per compression). VICTRE itself is unchanged.
# FEBio 3.8 misses an #include <limits> for GCC 11 (Ubuntu 22.04): added with -include, sources untouched.
#
#   docker build -t breastsim/victre-febio src/docker/victre_febio
FROM breastsim/victre

RUN sed -i 's/^\(deb .* jammy\(-updates\)\? main restricted\)$/\1 multiverse/' /etc/apt/sources.list \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libmkl-dev libmkl-full-dev libiomp-dev \
    && rm -rf /var/lib/apt/lists/*

ARG FEBIO_TAG=v3.8
RUN git clone --depth 1 --branch ${FEBIO_TAG} https://github.com/febiosoftware/FEBio /opt/FEBio \
    && cd /opt/FEBio && mkdir -p cbuild && cd cbuild \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-include limits" \
        -DUSE_MKL=ON -DMKL_INC=/usr/include/mkl -DMKL_LIB_DIR=/usr/lib/x86_64-linux-gnu \
        -DMKL_OMP_LIB=/usr/lib/x86_64-linux-gnu/libiomp5.so \
    && make -j"$(nproc)" \
    && chmod -R a+rX /opt/FEBio

ENV FEBIO_PATH=/opt/FEBio/cbuild/bin/febio3
