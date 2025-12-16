ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=11.8.0
ARG CUDNN_MAJOR_VERSION=8

########################
# Stage 1: build stage #
########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

ARG USER_UID=1001
ARG USER_GID=1001

# ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
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
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    python3-pip python3-dev python-is-python3 \
    && mkdir /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# clone & install relevant repositories
RUN git clone https://github.com/prov-gigapath/prov-gigapath.git /home/user/prov-gigapath

WORKDIR /opt/app/

# you can add any Python dependencies to requirements.in
RUN python -m pip install --upgrade pip setuptools pip-tools \
    && rm -rf /home/user/.cache/pip

# install slide2vec
COPY --chown=user:user requirements.in /opt/app/requirements.in
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.in \
    && rm -rf /home/user/.cache/pip

COPY --chown=user:user . /opt/app/
RUN python -m pip install /opt/app
RUN python -m pip install 'flash-attn>=2.7.1,<=2.8.0' --no-build-isolation


##########################
# Stage 2: runtime stage #
##########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn${CUDNN_MAJOR_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

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
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    python3-pip python3-dev python-is-python3 \
    && mkdir /var/run/sshd \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

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

# copy app code, and prov-gigapath
COPY --from=build /opt/app /opt/app
COPY --from=build /home/user/prov-gigapath /home/user/prov-gigapath

# add folders to python path (same as before)
ENV PYTHONPATH="/home/user/prov-gigapath:/home/user/CONCH:/home/user/MUSK:$PYTHONPATH"

# expose port for ssh and jupyter
EXPOSE 22 8888

WORKDIR /opt/app/

# switch to user
USER user
