FROM ubuntu:24.04

# Toolchain for building SimSYCL and this library. Boost::context is a hard
# requirement of SimSYCL; its remaining dependencies (libenvpp, nlohmann_json,
# fmt, Catch2) are fetched by CMake at configure time, so the image build needs
# network access.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libboost-context-dev \
      libboost-dev \
      ninja-build \
  && rm -rf /var/lib/apt/lists/*

# The SYCL implementation used by CI. Pinned to the commit this image was
# verified against; build with --build-arg SIMSYCL_REF=master to track upstream.
ARG SIMSYCL_REF=7053164fb663d554f2db734989cf8c2ca78f3877

# Build, self-test and install into /opt/simsycl, then drop the ~150 MiB source
# and build tree -- only the ~7 MiB install prefix is needed at runtime.
RUN git clone https://github.com/celerity/simsycl.git /tmp/simsycl \
  && git -C /tmp/simsycl checkout --detach "${SIMSYCL_REF}" \
  && cmake -S /tmp/simsycl -B /tmp/simsycl/build -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/opt/simsycl \
  && cmake --build /tmp/simsycl/build \
  && ctest --test-dir /tmp/simsycl/build --output-on-failure \
  && cmake --install /tmp/simsycl/build \
  && rm -rf /tmp/simsycl

# Let `find_package(SimSYCL CONFIG REQUIRED)` resolve without having to pass
# -DCMAKE_PREFIX_PATH=/opt/simsycl on every configure.
ENV CMAKE_PREFIX_PATH=/opt/simsycl

# The workspace is bind-mounted from the host (owned by uid 1000) while the
# container runs as root, which otherwise trips git's ownership check.
RUN git config --global --add safe.directory /workspace
