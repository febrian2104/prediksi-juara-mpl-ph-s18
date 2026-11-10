# Prediksi Juara MPL Philippines Season 18

Dashboard Streamlit berbahasa Indonesia untuk probabilitas kemenangan pertandingan,
peluang playoff, peluang juara, dan perbandingan dengan hasil aktual.

Batas informasi kompetisi: **9 November 2026**. Pengambilan tambahan dicatat dengan tanggal
sebenarnya di manifest sumber. S18 sudah selesai saat proyek dibuat: seluruh prediksinya
berlabel **rekonstruksi**, bukan prediksi live yang pernah disimpan sebelum pertandingan.

## Menjalankan

Gunakan Python 3.11+ dan jalankan:

```bash
make run
```

Perintah ini otomatis menyiapkan virtual environment dan dependensi melalui pip, lalu
menjalankan aplikasi di http://127.0.0.1:8501. Port lain: `make run PORT=8502`.
Dashboard membaca artefak yang disertakan; membuka halaman tidak menjalankan training.

## Fitur dan data

- Tiga halaman: Regular Season (minggu terbaru dahulu), Playoff, dan Hasil Prediksi.
- Katalog S1–S18 dengan struktur dan kontrak variabel mengikuti proyek referensi MPL ID.
- S6–S18: 852 laga canonical. S1–S5 masih parsial dan tidak dipakai untuk training.
- S5: 51 hasil terkumpul (41 regular season, 10 playoff/final). Empat pasangan regular
  belum terkumpul; tanggal/jam yang tidak diketahui tidak diisi. Laporan sekunder yang
  telah ditinjau bukan pengganti arsip resmi lengkap.
- Elo, Logistic Regression, Random Forest, dan XGBoost dibandingkan melalui walk-forward
  S10–S17. Logistic Regression dipilih dari evaluasi, bukan asumsi keunggulan model.
- Fitur hanya memakai riwayat sebelum laga; statistik/roster yang waktunya belum
  terverifikasi tidak menjadi fitur. Rating dan riwayat direset antarmusim sebelum S8
  dan pada batas awal franchise S8. Akuisisi tidak otomatis menggabungkan kode tim.
- Simulasi regular season dan bracket khusus PH S18; iterasi di `config/model.json`.
  Tiebreak tiga tim atau lebih masih memakai undian urutan awal saat durasi tidak tersedia.
- Prediksi lama diarsipkan dengan checksum; pembaruan tidak menimpa snapshot sebelumnya.

Sumber utama: [rulebook](https://ph-mpl.com/rules),
[jadwal/hasil](https://ph-mpl.com/schedule), [klasemen](https://ph-mpl.com/data),
dan [tim](https://ph-mpl.com/teams). Daftar sumber/tanggal pengambilan ada di
`data/raw/*source_manifest.csv`; cakupan tiap musim di `reports/season_coverage.csv`.
Bukti hasil awal dan kekurangannya: `reports/early_results_audit.json`.
Konflik penghitung S6/S8/S9/S16 belum tuntas; scope S10 cocok dengan laga termasuk playoff,
bukan regular season. S5 juga belum direkonsiliasi penuh.

Rekonstruksi regular season S18: accuracy 83,9%, log loss 0,436, Brier 0,139.
Pilihan juara pramusim Team Liquid PH (81,2%) **meleset**; juara aktual Aurora Gaming PH.
Rincian: `reports/s18_evaluation.json` dan `data/predictions/walk_forward_metrics.csv`.

## Pembaruan dan pengujian

```bash
make update         # Ambil sumber resmi dan bangun ulang artefak
make rebuild        # Bangun ulang memakai data lokal, tanpa web
make check          # Lint dan pengujian
make smoke          # Uji tiga halaman tanpa jaringan/training
make verify-history # Periksa checksum arsip prediksi
make verify-release RELEASE_ARGS=--fresh-env
make help
```

`make collect-history` memeriksa dan mengekspor bukti awal tanpa training. Untuk memakai
bukti lokal saja: `make collect-history HISTORY_ARGS=--cache-only`.
`make collect-details` mengambil scoreboard;
`make collect-season-data` mengambil statistik arsip; keduanya tidak melakukan training.
Prediksi mendatang: `make forecast SCHEDULE=path.csv PERIOD=label RULES=config/competition.json`.
Evaluasi terpisah: `make evaluate SNAPSHOT=path`. Aturan musim lain harus diverifikasi dahulu.

Untuk Streamlit Cloud, pilih `streamlit_app.py` dan pasang `requirements.txt`.
Uji rilis memakai salinan Git-visible dan venv baru, bukan bukti deployment ke akun Cloud.
Dokumentasi rinci di `docs/` dan `PROJECT_STATUS.md` hanya lokal, sengaja tidak disertakan
ke GitHub dan tidak diperlukan untuk menjalankan aplikasi atau pengujian rilis.

## Screenshot asli

Diambil langsung dari aplikasi lokal dengan artefak proyek ini.

| Regular Season | Playoff | Hasil Prediksi |
| :---: | :---: | :---: |
| ![Regular Season](assets/screenshots/regular-season.png) | ![Playoff](assets/screenshots/playoff.png) | ![Hasil Prediksi](assets/screenshots/hasil-prediksi.png) |

