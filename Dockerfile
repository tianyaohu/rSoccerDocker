# ---- Stage 1: build wheels ----
    FROM mambaorg/micromamba:1.5.10 AS builder

    USER root
    RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        git \
        libode-dev \
     && rm -rf /var/lib/apt/lists/*
    
    USER mambauser
    SHELL ["/bin/bash", "-lc"]
    
    RUN micromamba config append channels conda-forge && \
        micromamba config set channel_priority strict
    
    RUN micromamba create -y -n rsoccer310 -c conda-forge \
        python=3.10 pip cmake ninja && \
        micromamba clean -a -y
    
    ENV CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
    
    RUN micromamba run -n rsoccer310 python -m pip install -U pip wheel setuptools
    RUN mkdir -p /tmp/wheels
    RUN micromamba run -n rsoccer310 python -m pip wheel --wheel-dir /tmp/wheels rc-robosim
    RUN micromamba run -n rsoccer310 python -m pip wheel --wheel-dir /tmp/wheels \
        "git+https://github.com/robocin/rSoccer.git@3adf7c3e89fe6d1b47431ee4099dc5ee89420415#egg=rsoccer-gym"
    
    # ---- Stage 2: runtime ----
    FROM mambaorg/micromamba:1.5.10 AS runtime
    
    USER root
    RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates \
        # rSoccer physics engine
        libode8 \
        # X11 client libs — needed for --render via X11 forwarding
        libx11-6 libxext6 libxrender1 libxi6 \
        # Mesa software OpenGL (no GPU required)
        libgl1-mesa-glx libglib2.0-0 \
        # SDL2 — used by pygame/rSoccer renderer
        libsdl2-2.0-0 libsdl2-image-2.0-0 \
     && rm -rf /var/lib/apt/lists/*
    
    # Headless-safe defaults (avoid ALSA + XDG warnings inside Docker)
    ENV XDG_RUNTIME_DIR=/tmp/xdg
    ENV SDL_AUDIODRIVER=dummy
    RUN mkdir -p /tmp/xdg && chmod 700 /tmp/xdg
    
    # # Copy and register the entrypoint (must be done as root)
    # COPY entrypoint.sh /usr/local/bin/entrypoint.sh
    # RUN chmod +x /usr/local/bin/entrypoint.sh
    
    USER mambauser
    SHELL ["/bin/bash", "-lc"]
    
    RUN micromamba config append channels conda-forge && \
        micromamba config set channel_priority strict
    
    RUN micromamba create -y -n rsoccer310 -c conda-forge \
        python=3.10 pip && \
        micromamba clean -a -y
    
    # Install pre-built rSoccer wheels
    COPY --from=builder /tmp/wheels /wheels
    RUN micromamba run -n rsoccer310 python -m pip install --no-cache-dir /wheels/*.whl
    
    # Python runtime dependencies
    RUN micromamba run -n rsoccer310 python -m pip install --no-cache-dir \
        gymnasium \
        stable-baselines3 \
        pytest \
        wandb \
        pygame \
        moviepy imageio imageio-ffmpeg \
        tensorboard \
        tqdm \
        rich \
     && micromamba run -n rsoccer310 python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
    
    # Port for TensorBoard
    EXPOSE 6006
    
    # tell micromamba-docker which environment to activate
    ENV ENV_NAME=rsoccer310
    
