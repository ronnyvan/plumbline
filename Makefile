# Plumbline. Every target is a thin wrapper; the logic lives in Python so it runs the same in CI.

ifeq ($(OS),Windows_NT)
  VENV_PY := .venv/Scripts/python.exe
else
  VENV_PY := .venv/bin/python
endif
PY ?= $(VENV_PY)
BOOTSTRAP_PY ?= python

.PHONY: help venv install install-pose test test-pose lint format eval eval-fast verify report \
        check-leakage golden analyze clean-artifacts

help:
	@echo "install        create .venv and install runtime + dev dependencies"
	@echo "install-pose   add RTMPose (GPU) dependencies"
	@echo "test           unit, property and golden tests (no GPU)"
	@echo "test-pose      tests against the real RTMPose backend (GPU)"
	@echo "lint           ruff"
	@echo "eval           full evaluation -> eval/results/eval.json (commit it)"
	@echo "eval-fast      small corpus, 1k bootstrap resamples -> eval/results/eval-fast.json"
	@echo "verify         regenerate results and fail unless they match the committed file"
	@echo "report         eval/results/eval.json -> eval/results/REPORT.md"
	@echo "check-leakage  fail if any subject appears in more than one split"
	@echo "golden         regenerate golden reports (review the diff!)"
	@echo "analyze        VIDEO=clip.mp4 EXERCISE=lat_pulldown STATION=S-LAT"

venv:
	$(BOOTSTRAP_PY) -m venv .venv

install: venv
	$(VENV_PY) -m pip install --upgrade pip
	$(VENV_PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-pose:
	$(PY) -m pip install -r requirements-pose.txt
	$(PY) -m pip install --no-deps rtmlib==0.0.16

test:
	$(PY) -m pytest -q

test-pose:
	$(PY) -m pytest -q -m pose

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff check --fix .

eval:
	$(PY) -m eval.harness.run eval

eval-fast:
	$(PY) -m eval.harness.run eval --fast

verify:
	$(PY) -m eval.harness.run verify

report:
	$(PY) -m eval.harness.run report

check-leakage:
	$(PY) -m eval.harness.run check-leakage

golden:
	$(PY) -m tests.golden.cases

analyze:
	$(PY) -m core.cli analyze "$(VIDEO)" --exercise $(EXERCISE) --station $(STATION)

clean-artifacts:
	$(PY) -c "import shutil; shutil.rmtree('artifacts', ignore_errors=True)"
