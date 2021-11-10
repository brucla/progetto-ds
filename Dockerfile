FROM python:3.8

ENV HOME /root
ENV PYTHONPATH "/usr/lib/python3/dist-packages:/usr/local/lib/python3.5/site-packages"

WORKDIR /progetto

# Install dependencies
RUN apt update \
    && apt upgrade -y \
    && apt autoremove -y \
    && apt install -y \
        gcc \
        build-essential \
        zlib1g-dev \
        wget \
        unzip \
        cmake \
        python3-dev \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
	libjemalloc-dev\
	libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-regex-dev \
        python-dev \
        autoconf \
        flex \
        bison \
    && apt clean

# Install Python packages
RUN pip install --upgrade pip \
    && pip install \
        ipython[all] \
        numpy \
        pandas \
        scipy \
	scikit-learn \
	plotly \
	statsmodels \
	streamlit\
    && rm -fr /root/.cache

COPY . .

CMD streamlit run progetto.py