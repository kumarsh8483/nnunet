FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

EXPOSE 4321/tcp

ENV TZ=America/Chicago \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    nnUNet_n_proc_DA=32 \
    nnUNet_raw="/input" \
    nnUNet_preprocessed="/preprocessed" \
    nnUNet_master_port=4321 \
    nnUNet_results="/output" \
    HDF5_USE_FILE_LOCKING=FALSE 

WORKDIR /root

RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && \
    echo ${TZ} > /etc/timezone && \
    apt-get remove -y linux-libc-dev && \ 
    apt-get install -y \
        git \
        curl \
        graphviz \
        libx11-6 \
        ca-certificates && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /input /output /preprocessed

RUN conda update -y conda && \
    python -m pip install -U pip && \
    python -m pip install nnunet && \
    python -m pip install -U git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer && \
    conda clean -ya && \
    rm -rf $(python -m pip cache dir)
