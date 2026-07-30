# Workshop 5.4 — Multimodal Edge AI for Autonomous Response
#
# CPU-only base. 5.4 trains a ~125KB MLP on 500 synthetic rows generated in the
# notebooks, so there is nothing here that benefits from a GPU (the notebooks
# already fall back with torch.device('cuda' if available else 'cpu')). Using the
# CPU tag instead of cuda12 keeps the image small and lets this run on the
# user-cpu-large nodegroup.
FROM quay.io/jupyter/pytorch-notebook:python-3.11

ARG JUPYTERHUB_VERSION=5.4.4

# PyTorch/OpenMP size their thread pools from the HOST core count (16 on
# m3.large), not the container's cpu_limit. With ~10 users per node that means
# ~160 threads fighting over 16 cores. Pin these to match cpu_limit in the hub
# profile so concurrent training stays responsive.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

COPY ./requirements.txt .

RUN pip3 install --no-cache -r requirements.txt

RUN pip3 install --no-cache \
    jupyterhub==$JUPYTERHUB_VERSION

RUN pip3 install --no-cache \
    nbgitpuller
