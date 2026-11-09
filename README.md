# Prediksi Juara MPL Philippines Season 18

Proyek ini memperkirakan probabilitas kemenangan pertandingan, peluang lolos playoff,
dan peluang juara MPL Philippines Season 18. Dashboard Streamlit membandingkan hasil
rekonstruksi prediksi dengan hasil aktual secara ringkas dan berbahasa Indonesia.

Data dibatasi sampai **9 November 2026**. Karena S18 sudah selesai ketika proyek dibuat,
prediksi S18 diberi label **`reconstructed`**. Prediksi tersebut membangun ulang keadaan
sebelum setiap laga, tetapi bukan arsip prediksi yang benar-benar disimpan pada tanggal
pertandingan.

## Tampilan aplikasi

Screenshot berikut diambil langsung dari aplikasi lokal yang menggunakan artefak prediksi
di repository ini.

| Regular Season | Playoff | Hasil Prediksi |
| :---: | :---: | :---: |
| [![Regular Season](assets/screenshots/regular-season.png)](assets/screenshots/regular-season.png) | [![Playoff](assets/screenshots/playoff.png)](assets/screenshots/playoff.png) | [![Hasil Prediksi](assets/screenshots/hasil-prediksi.png)](assets/screenshots/hasil-prediksi.png) |

## Fitur

- **Regular Season:** klasemen resmi, perkembangan peluang playoff, jadwal, probabilitas,
  skor aktual, dan status tepat/meleset; minggu terbaru ditampilkan lebih dahulu.
- **Playoff:** bagan hybrid elimination resmi, peluang juara sebelum playoff, prediksi
  setiap seri, dan hasil aktual.
- **Hasil Prediksi:** pilihan juara pramusim dan sebelum playoff, hasil akhir, evaluasi S18,
  serta perbandingan Elo, Logistic Regression, Random Forest, dan XGBoost.
- Pipeline memisahkan `data/raw`, `data/clean`, `data/features`, dan `data/predictions`.
- Semua fitur pertandingan dihitung dari informasi sebelum waktu laga. Audit leakage saat
  ini menghasilkan nol pelanggaran.
- Training dan simulasi dijalankan melalui perintah pembaruan, bukan saat dashboard dibuka.

## Data dan format resmi

Sumber utama adalah [rulebook resmi MPL PH S18](https://ph-mpl.com/rules),
[jadwal dan hasil resmi](https://ph-mpl.com/schedule),
[klasemen resmi](https://ph-mpl.com/data), [daftar tim resmi](https://ph-mpl.com/teams),
halaman roster resmi masing-masing tim, serta
[rilis juara dari MOONTON Games](https://en.moonton.com/news/391.html). Seluruh sumber
dicatat di `data/raw/source_manifest.csv`; pemeriksaan sumber awal dicatat di
`data/raw/historical_source_manifest.csv`, dengan tanggal akses 9 November 2026.

Format yang diterapkan:

- delapan tim franchise, double round-robin, BO3, 14 pertandingan per tim;
- satu poin untuk kemenangan seri dan nol untuk kekalahan;
- tiebreaker: net game wins, head-to-head match wins, head-to-head net game wins, lalu
  BO1 resmi bila masih diperlukan;
- enam tim ke playoff; seed 3–6 masuk play-in, seed 1–2 mendapat bye;
- seluruh seri playoff BO5, kecuali lower bracket final dan grand final yang BO7.

Katalog mencakup **Season 1–18**, dengan status yang berbeda per musim:

- **S1–S5:** 14 hasil parsial dari berita resmi lama, laporan pertandingan, dan sumber
  pembanding. Jam mulai belum lengkap; hasil BO2 seri S1 dipertahankan. Belum masuk model.
- **S6–S18:** 852 laga canonical, termasuk 788 laga sebelum S18 dan 64 laga S18.
  S8 dipulihkan dari halaman jadwal resmi. Tiga skor S6 yang rusak dikoreksi memakai
  dua laporan pertandingan dan berita resmi liga; skor asli serta sumber disimpan.
- Roster S18 memuat 49 pemain dan 17 pelatih/staf. Ada tujuh observasi pemain historis,
  tetapi tanggal efektifnya belum cukup lengkap untuk fitur roster.
- Tujuh tabel mengikuti nama kolom dataset referensi Indonesia. Statistik 49 pemain S18
  tersedia sebagai observasi pascamusim; kolom yang tidak tersedia tetap null dan statistik
  tersebut tidak digunakan untuk fitur S18. [Kamus variabel](docs/DATA_DICTIONARY.md)
  menjelaskan pemetaan dan tipe data.

Cakupan dan kekurangan setiap musim tercatat di `reports/season_coverage.csv`.
Data per musim memakai `data/mpl-season1` sampai `data/mpl-season17` dan `data/season18`,
mengikuti proyek referensi MPL Indonesia. Data olahan juga tersedia di
`data/interim/normalized` dan `data/processed/{canonical,features,predictions}`;
model berada di `artifacts/final_match_model.joblib`.
[Pemetaan struktur](docs/STRUCTURE.md) dan [catatan sumber](docs/DATA_SOURCES.md) menjelaskan
penyesuaian khusus PH. Tabel opsional yang belum tersedia hanya berisi header, tanpa data buatan.

## Hasil model

Walk-forward backtesting memakai S10–S17 sebagai jendela uji dan hanya melatih pada musim
yang lebih lama. Setelah S8 ditambahkan dan alias historis ECHO dibetulkan,
Logistic Regression menjadi model produksi dengan log loss 0,613, dibanding Elo 0,617,
Random Forest 0,623, dan XGBoost 0,625. Pemilihan ini dilakukan sebelum evaluasi S18 dan hasil S18 tidak dipakai untuk
training atau pemilihan model.

Pada rekonstruksi regular season S18, model memperoleh accuracy 82,1%, log loss 0,436, dan
Brier score 0,139. Rekonstruksi pramusim memilih Team Liquid PH dengan peluang juara 78,9%,
sedangkan juara aktual adalah Aurora Gaming PH. Hasil meleset ini tetap ditampilkan apa
adanya.

## Menjalankan dashboard

Gunakan Python 3.11 atau lebih baru. Cukup jalankan:

```bash
make run
```

Pada pemakaian pertama, perintah tersebut otomatis membuat `.venv` dan memasang dependensi
dengan `pip` (bukan `uv`), lalu membuka dashboard di `http://127.0.0.1:8501`. Pemakaian
berikutnya langsung menjalankan aplikasi. Dashboard hanya membaca artefak CSV yang sudah
tersedia, sehingga dapat dijalankan tanpa training ulang.

Target lain yang tersedia dapat dilihat dengan `make help`. Port dapat diganti, misalnya
`make run PORT=8502`.

## Memperbarui data dan prediksi

Pembaruan penuh dari sumber resmi:

```bash
make update
```

Membangun ulang dari data mentah lokal tanpa mengakses web:

```bash
make rebuild
```

Jumlah simulasi dapat diubah melalui `simulation.iterations` di `config/model.json`.
Setiap pembaruan otomatis mengarsipkan keadaan sebelum dan sesudah pipeline di
`data/predictions/history`. Snapshot memakai waktu simpan sebenarnya, checksum, dan pembuatan
eksklusif agar riwayat tidak tertimpa. Jalankan `make verify-history` untuk memeriksa integritas.
Label S18 tetap rekonstruksi. [Petunjuk prediksi live](docs/LIVE_PREDICTIONS.md) menyediakan
perintah untuk laga mendatang dan evaluasi terpisah; saat ini belum ada prediksi live asli.

BO1 tiebreak disimulasikan memakai probabilitas model. Untuk tiebreak tiga tim atau lebih,
urutan awal masih menggunakan undian karena total durasi kemenangan tidak tersedia.
Snapshot mingguan memakai siang hari berikutnya sebagai asumsi rekonstruksi; arsip tidak
menyediakan waktu publikasi hasil yang sebenarnya. Keterbatasan ini juga tampil di dashboard.

## Pengujian

```bash
make check
```

Untuk mengambil ulang screenshot setelah dashboard lokal berjalan:

```bash
.venv/bin/python scripts/capture_screenshots.py
```

## Streamlit Cloud

Gunakan `streamlit_app.py` sebagai main file. `requirements.txt` hanya berisi dependensi
runtime dan seluruh file prediksi yang dibaca dashboard sudah disertakan. Aplikasi tidak
menjalankan kolektor, training, atau simulasi ketika halaman dibuka.

Daftar pekerjaan selesai dan tertunda: [PROJECT_STATUS.md](PROJECT_STATUS.md).
Operasional lokal: [docs/OPERATIONS.md](docs/OPERATIONS.md).

