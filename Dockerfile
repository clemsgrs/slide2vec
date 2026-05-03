ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=12.8.1

########################
# Stage 1: build stage #
########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build

ARG USER_UID=1001
ARG USER_GID=1001

# ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam

USER root

# create user and I/O dirs
RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user \
    && mkdir /input /output \
    && chown user:user /input /output

# set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    cmake \
    zlib1g-dev \
    libnuma1 \
    libspatialindex-dev \
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    python3 \
    python3-dev \
    python3-venv \
    python3-distutils \
    python3-pip \
    && mkdir /var/run/sshd \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# libjpeg-turbo 3.x (required by PyTurboJPEG>=2)
ARG LIBJPEG_TURBO_VERSION=3.1.0
RUN curl -fsSL https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/${LIBJPEG_TURBO_VERSION}/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz \
      | tar xz -C /tmp \
    && cd /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION} \
    && cmake -G"Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local . \
    && make -j"$(nproc)" && make install \
    && ldconfig \
    && rm -rf /tmp/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}

WORKDIR /opt/app/

ARG PYTORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG GIT_MODEL_DEPENDENCIES="git+https://github.com/lilab-stanford/MUSK.git git+https://github.com/Mahmoodlab/CONCH.git git+https://github.com/prov-gigapath/prov-gigapath.git git+https://github.com/facebookresearch/sam2.git"

RUN python -m pip install --upgrade pip setuptools pip-tools \
    && python -m pip install hatchling psutil \
    && rm -rf /home/user/.cache/pip

# install slide2vec with all model extras
COPY --chown=user:user slide2vec /opt/app/slide2vec
COPY --chown=user:user pyproject.toml /opt/app/pyproject.toml
COPY --chown=user:user README.md /opt/app/README.md
COPY --chown=user:user LICENSE /opt/app/LICENSE
RUN printf '%s\n' \
    'torch' \
    > /opt/app/constraints-cu128.txt

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    "/opt/app[fm]"

RUN python -m pip install --no-cache-dir --no-color \
    -c /opt/app/constraints-cu128.txt \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    ${GIT_MODEL_DEPENDENCIES}

RUN python -m pip install \
    --no-cache-dir \
    --extra-index-url "${PYTORCH_CUDA_INDEX_URL}" \
    -c /opt/app/constraints-cu128.txt \
    'flash-attn>=2.7.1,<=2.8.0' \
    --no-build-isolation \
    && rm -rf /home/user/.cache/pip

##########################
# Stage 2: runtime stage #
##########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ARG USER_UID=1001
ARG USER_GID=1001

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Europe/Amsterdam

USER root

RUN groupadd --gid ${USER_GID} user \
    && useradd -m --no-log-init --uid ${USER_UID} --gid ${USER_GID} user

# create input/output directory
RUN mkdir /input /output && \
    chown user:user /input /output

# set /home/user as working directory
WORKDIR /home/user
ENV PATH="/home/user/.local/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtiff-dev \
    zlib1g-dev \
    libnuma1 \
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    python3 \
    python3-venv \
    python3-distutils \
    python3-pip \
    && mkdir /var/run/sshd \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# libjpeg-turbo 3.x (copied from build stage)
COPY --from=build /usr/local/lib/libjpeg* /usr/local/lib/
COPY --from=build /usr/local/lib/libturbojpeg* /usr/local/lib/
RUN ldconfig

# install ASAP
ARG ASAP_URL=https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb
RUN apt-get update && curl -L ${ASAP_URL} -o /tmp/ASAP.deb && apt-get install --assume-yes /tmp/ASAP.deb && \
    SITE_PACKAGES=`python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"` && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth" && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# copy Python libs & entrypoints from build stage (includes flash-attn, your deps, ASAP .pth)
COPY --from=build /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=build /usr/local/bin /usr/local/bin

# register libnvimgcodec so cucim can use GPU-accelerated JPEG decoding
RUN echo "/usr/local/lib/python3.10/dist-packages/nvidia/nvimgcodec" > /etc/ld.so.conf.d/nvimgcodec.conf && \
    ldconfig

# copy app code
COPY --from=build /opt/app /opt/app

# expose port for ssh and jupyter
EXPOSE 22 8888

WORKDIR /opt/app/

# switch to user
USER user
