.DEFAULT_GOAL := run

SYSTEM_PYTHON ?= python3
VENV := .venv
PYTHON := $(VENV)/bin/python
RUNTIME_STAMP := $(VENV)/.runtime-ready
DEV_STAMP := $(VENV)/.dev-ready
HOST ?= 127.0.0.1
PORT ?= 8501

.PHONY: run setup dev-setup update rebuild test lint check help archive verify-history forecast evaluate collect-history collect-details collect-season-data audit-game-data dashboard verify

run: setup
	$(PYTHON) -m streamlit run streamlit_app.py --server.address $(HOST) --server.port $(PORT)

dashboard: run

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

verify: check verify-history

collect-history: setup
	$(PYTHON) scripts/collect_history.py

collect-details: setup
	$(PYTHON) scripts/collect_game_details.py $(DETAIL_ARGS)

collect-season-data: setup
	$(PYTHON) scripts/collect_season_evidence.py $(EVIDENCE_ARGS)

audit-game-data: setup
	$(PYTHON) scripts/audit_game_data.py

archive: setup
	$(PYTHON) scripts/update_project.py --archive-only

verify-history: setup
	$(PYTHON) scripts/update_project.py --verify-history

forecast: setup
	@test -n "$(SCHEDULE)" -a -n "$(PERIOD)" || (echo 'Isi SCHEDULE dan PERIOD; lihat docs/LIVE_PREDICTIONS.md.'; exit 2)
	$(PYTHON) scripts/forecast.py save --schedule "$(SCHEDULE)" --period "$(PERIOD)" --rules "$(or $(RULES),config/competition.json)"

evaluate: setup
	@test -n "$(SNAPSHOT)" || (echo 'Isi SNAPSHOT dengan jalur prediksi live.'; exit 2)
	$(PYTHON) scripts/forecast.py evaluate --snapshot "$(SNAPSHOT)"

help:
	@printf '%s\n' \
		'make run       Siapkan environment otomatis lalu jalankan dashboard' \
		'make setup     Siapkan virtual environment dan dependensi runtime' \
		'make update    Ambil data resmi lalu perbarui seluruh artefak' \
		'make rebuild   Bangun ulang artefak dari data mentah lokal' \
		'make check     Jalankan lint dan seluruh pengujian' \
		'make archive   Simpan arsip artefak saat ini tanpa menimpa riwayat' \
		'make collect-history  Periksa ulang sumber historis Season 1-5' \
		'make collect-details  Ambil scoreboard game resmi tanpa training ulang' \
		'make collect-season-data  Ambil statistik arsip dan audit silang klasemen' \
		'make audit-game-data  Bandingkan ringkasan game dan scoreboard lokal' \
		'make verify-history  Periksa integritas seluruh snapshot' \
		'make forecast SCHEDULE=... PERIOD=... RULES=...  Simpan prediksi mendatang' \
		'make evaluate SNAPSHOT=...  Evaluasi prediksi live secara terpisah' \
		'make help      Tampilkan bantuan ini'
