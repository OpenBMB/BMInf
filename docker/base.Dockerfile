FROM nvidia/cuda:10.1-runtime
WORKDIR /build
ENV LANG=C.UTF-8
ENV CUDA_VERSION=101
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-wheel libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install setuptools
ADD . .
RUN python3 setup.py sdist --formats=gztar
RUN pip3 install ./dist/*.tar.gz
WORKDIR /root
RUN rm -r /build
ADD examples examples