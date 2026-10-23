# breastsim/victre + FEBio, the finite element solver VICTRE's compression step (breastCompress)
# calls through $FEBIO_PATH. VICTRE documents FEBio 2.x (binary download, account required) and
# says 3.x "might also work but has not been tested"; this image builds FEBio 3.8 from the official
# open-source repository (MIT), without MKL (Skyline linear solver). VICTRE itself is unchanged.
# FEBio 3.8 misses an #include <limits> for GCC 11 (Ubuntu 22.04): added with -include, sources untouched.
#
#   docker build -t breastsim/victre-febio src/docker/victre_febio
FROM breastsim/victre

ARG FEBIO_TAG=v3.8
RUN git clone --depth 1 --branch ${FEBIO_TAG} https://github.com/febiosoftware/FEBio /opt/FEBio \
    && cd /opt/FEBio && mkdir -p cbuild && cd cbuild \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-include limits" && make -j"$(nproc)" \
    && chmod -R a+rX /opt/FEBio

ENV FEBIO_PATH=/opt/FEBio/cbuild/bin/febio3
