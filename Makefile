################################### HEADER ####################################

SHELL := /bin/bash
MAKEFLAGS += --always-make

.ONESHELL:

include .env
export $(shell sed 's/=.*//' .env)

#################################### BUILD ####################################

COMPETITION := santa-2022
CONDA_ENV := kaggle-${COMPETITION}
PYTHON_VER := 3.10.0

DATA_DIR := $(abspath data)
OUTPUT_DIR := ${abspath output}
SWEEPA_DIR := $(abspath sweeps)

build: conda-env data

conda-env: conda-env-base
	conda install -y -n ${CONDA_ENV} -c conda-forge \
		grpcio=1.43.0
	conda run -n ${CONDA_ENV} pip install \
		pip install "ray[tune]"==2.2.0

conda-env-base:
	if ! { conda env list | grep ${CONDA_ENV}; } >/dev/null 2>&1; then
		conda create -y -n ${CONDA_ENV}
	fi
	conda install -y -n ${CONDA_ENV} -c conda-forge \
		python==${PYTHON_VER} \
		matplotlib=3.6.2 \
		numpy==1.24.0 \
		pillow==9.2.0\
		jupyterlab=3.5.2
	conda run -n ${CONDA_ENV} pip install \
		kaggle==1.5.12

data:
	mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
	conda run -n ${CONDA_ENV} kaggle competitions download ${COMPETITION}
	unzip ${COMPETITION}.zip && rm -f ${COMPETITION}.zip

##################################### RUN #####################################

grid-search:
	python -u src/grid_search.py \
		--n-links ${N_LINKS} \
		--src-path ${DATA_DIR}/image.png \
		--output-dir ${OUTPUT_DIR} \
		--sweep ${SWEEP} ${FLAGS}

pbt:
	python -u src/pbt.py \
		--n-links ${N_LINKS} \
		--src-path ${DATA_DIR}/image.png \
		--output-dir ${OUTPUT_DIR} \
		--sweep ${SWEEP} ${FLAGS}

#################################### UTILS ####################################

test:
	echo "This is a test!"

print-%: ; @echo $* = $($*)
