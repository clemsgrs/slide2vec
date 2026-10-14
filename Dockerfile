# Release image: CUDA 12.8, the locked slide2vec environment in /opt/venv.
#
# The environment comes from uv.lock. SLIDE2VEC_EXTRAS and SLIDE2VEC_GROUPS
# select the extras and dependency groups:
#
#   docker build -t slide2vec .
#       cu128 torch, the `fm` extra, flash-attn (gigapath-slide needs it on
#       GPUs with compute capability above 7) and the git-installed encoder
#       packages (MUSK, CONCH, prov-gigapath, sam2).
#
#   docker build -t slide2vec:prism2 \
#       --build-arg SLIDE2VEC_EXTRAS="cu128 prism2" \
#       --build-arg SLIDE2VEC_GROUPS="" .
#       PRISM2 in its own environment (transformers<4.53 and flash-attn).
#
# Both images build flash-attn, which ships only an sdist that imports torch at
# build time. pyproject.toml declares torch (matching the locked runtime torch)
# as an extra build dependency of flash-attn, so a single `uv sync` builds it.
# If that fails, sync once without it, then build it against the installed
# torch (shown for the default image; use the prism2 flags for that variant):
#   uv sync --locked --no-install-project --extra cu128 --extra fm \
#       --extra flash-attn --group models-git --no-install-package flash-attn
#   uv pip install --python /opt/venv/bin/python setuptools wheel packaging psutil ninja
#   uv sync --locked --no-install-project --extra cu128 --extra fm \
#       --extra flash-attn --group models-git --no-build-isolation-package flash-attn

ARG UBUNTU_VERSION=22.04
ARG CUDA_MAJOR_VERSION=12.8.1
ARG PYTHON_VERSION=3.11

########################
# Stage 1: build stage #
########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS build

ARG USER_UID=1001
ARG USER_GID=1001
ARG PYTHON_VERSION

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
    software-properties-common \
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
    gnupg2 \
    gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    && mkdir /var/run/sshd \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
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

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ARG SLIDE2VEC_EXTRAS="cu128 fm flash-attn"
ARG SLIDE2VEC_GROUPS="models-git"

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=/usr/bin/python${PYTHON_VERSION} \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1

# locked third-party dependencies first, so this layer re-runs only when
# pyproject.toml or uv.lock changes
COPY --chown=user:user pyproject.toml uv.lock README.md /opt/app/
RUN set -eux; \
    flags=""; \
    for extra in ${SLIDE2VEC_EXTRAS}; do flags="${flags} --extra ${extra}"; done; \
    for group in ${SLIDE2VEC_GROUPS}; do flags="${flags} --group ${group}"; done; \
    uv sync --locked --no-install-project ${flags}

# then slide2vec itself
COPY --chown=user:user slide2vec /opt/app/slide2vec
COPY --chown=user:user LICENSE /opt/app/LICENSE
RUN uv pip install --python /opt/venv/bin/python --no-deps /opt/app

##########################
# Stage 2: runtime stage #
##########################
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ARG USER_UID=1001
ARG USER_GID=1001
ARG PYTHON_VERSION

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
    software-properties-common \
    libtiff-dev \
    zlib1g-dev \
    libnuma1 \
    libspatialindex-dev \
    curl \
    vim screen \
    zip unzip \
    git \
    openssh-server \
    gnupg2 \
    gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-distutils \
    && mkdir /var/run/sshd \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
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

# copy the environment (dependencies, slide2vec and its entrypoint) and uv
COPY --from=build /opt/venv /opt/venv
COPY --from=build /usr/local/bin/uv /usr/local/bin/uv
ENV PATH="/opt/venv/bin:${PATH}"

# make ASAP importable from the environment
RUN SITE_PACKAGES=$(/opt/venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    printf "/opt/ASAP/bin/\n" > "${SITE_PACKAGES}/asap.pth"

# register libnvimgcodec so cucim can use GPU-accelerated JPEG decoding
RUN echo "/opt/venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/nvimgcodec" > /etc/ld.so.conf.d/nvimgcodec.conf && \
    ldconfig

# copy app code
COPY --from=build /opt/app /opt/app

# expose port for ssh and jupyter
EXPOSE 22 8888

WORKDIR /opt/app/

# switch to user
USER user
