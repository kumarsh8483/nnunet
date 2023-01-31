FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04
USER root

EXPOSE 4321/tcp

ENV TZ=America/Chicago \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    nnUNet_n_proc_DA=32 \
    nnUNet_raw_data_base="/input" \
    nnUNet_preprocessed="/preprocessed" \
    nnUNet_master_port=4321 \
    RESULTS_FOLDER="/output" \
    HDF5_USE_FILE_LOCKING=FALSE 

WORKDIR /root

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime
RUN echo ${TZ} > /etc/timezone
RUN apt-get remove -y linux-libc-dev
RUN apt-get install -y git curl libx11-6 ca-certificates
RUN apt-get upgrade -y
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*
RUN mkdir -p /input /output /preprocessed

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN conda update -y conda
RUN pip install -U pip
RUN pip install nnunet
RUN pip install gdown
RUN pip install -U python-dateutil && \
    pip install -U git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer && \
    conda clean -ya && \
    rm -rf $(python -m pip cache dir)

RUN pip install matplotlib
    
