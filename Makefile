.DEFAULT_GOAL := run

SYSTEM_PYTHON ?= python3
VENV := .venv
PYTHON := $(VENV)/bin/python
RUNTIME_STAMP := $(VENV)/.runtime-ready
DEV_STAMP := $(VENV)/.dev-ready
HOST ?= 127.0.0.1
PORT ?= 8501

.PHONY: run setup dev-setup update rebuild test lint check help

run: setup
	$(PYTHON) -m streamlit run streamlit_app.py --server.address $(HOST) --server.port $(PORT)

setup: $(RUNTIME_STAMP)

dev-setup: $(DEV_STAMP)

$(PYTHON):
	$(SYSTEM_PYTHON) -m venv $(VENV)

$(RUNTIME_STAMP): requirements.txt $(PYTHON)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	touch $(RUNTIME_STAMP)

$(DEV_STAMP): requirements-dev.txt $(RUNTIME_STAMP)
	$(PYTHON) -m pip install -r requirements-dev.txt
	touch $(DEV_STAMP)

update: setup
	$(PYTHON) scripts/update_project.py

rebuild: setup
	$(PYTHON) scripts/update_project.py --no-fetch

test: dev-setup
	$(PYTHON) -m pytest -q

lint: dev-setup
	$(PYTHON) -m ruff check .

check: lint test

help:
	@printf '%s\n' \
		'make run       Siapkan environment otomatis lalu jalankan dashboard' \
		'make setup     Siapkan virtual environment dan dependensi runtime' \
		'make update    Ambil data resmi lalu perbarui seluruh artefak' \
		'make rebuild   Bangun ulang artefak dari data mentah lokal' \
		'make check     Jalankan lint dan seluruh pengujian' \
		'make help      Tampilkan bantuan ini'
