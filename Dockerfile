# Use CUDA 11.8 runtime image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh && \
    /bin/bash /root/miniconda.sh -b -p $CONDA_DIR && \
    rm /root/miniconda.sh && \
    conda clean --all -f -y

# Copy environment.yaml
COPY environment.yaml /tmp/environment.yaml

# Create conda environment
RUN conda env create -f /tmp/environment.yaml && \
    conda clean --all -f -y

# Set up entry point to activate conda environment
SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]

# Set working directory
WORKDIR /workspace

# Default command
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ml"]
CMD ["/bin/bash"]
