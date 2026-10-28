.PHONY: help setup lint test data train baseline evaluate export report all clean

PY ?= python

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## install the package and dev tooling
	$(PY) -m pip install -e ".[dev]"

lint:  ## static checks
	ruff check src tests

test:  ## unit tests
	pytest

data:  ## download the corpus and cache MFCC features (~2.3 GB download)
	kws data

train:  ## fit the DS-CNN
	kws train --arch dscnn

baseline:  ## fit the linear baseline
	kws train --arch linear

evaluate:  ## score every trained model on the test split
	kws evaluate

export:  ## ONNX export, int8 quantisation, size and latency budget
	kws export

report:  ## collect metrics and figures into reports/
	kws report

all: data train baseline evaluate export report  ## full reproduction

clean:
	rm -rf models reports/*.json reports/*.png reports/RESULTS.md data/cache
