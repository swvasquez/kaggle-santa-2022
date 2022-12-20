################################### HEADER ####################################

SHELL := /bin/bash
MAKEFLAGS += --always-make

.ONESHELL:

#################################### BUILD ####################################

COMPETITION := santa-2022
CONDA_ENV := kaggle-${COMPETITION}
PYTHON_VER := 3.11.0

DATA_DIR := $(abspath data)

build: conda-env data

conda-env: conda-env-base

conda-env-base:
	if ! { conda env list | grep ${CONDA_ENV}; } >/dev/null 2>&1; then
		conda create -y -n ${CONDA_ENV}
	fi
	conda install -y -n ${CONDA_ENV} -c conda-forge \
		python==${PYTHON_VER} \
		jupyterlab==3.5.1
	conda run -n ${CONDA_ENV} pip install \
		kaggle==1.5.12 \
		numpy==1.24.0

data:
	mkdir -p ${DATA_DIR} && cd ${DATA_DIR}
	conda run -n ${CONDA_ENV} kaggle competitions download ${COMPETITION}
	unzip ${COMPETITION}.zip && rm -f ${COMPETITION}.zip

#################################### UTILS ####################################

test:
	echo "This is a test!"

print-%: ; @echo $* = $($*)

###############################################################################
