# TFX Adult Pipeline – Income Classification

Proyek ini merupakan implementasi *machine learning pipeline* menggunakan **TensorFlow Extended (TFX)** untuk memprediksi pendapatan individu berdasarkan data demografis (Adult Income Dataset). Pipeline dijalankan dengan `InteractiveContext` dan mencakup keseluruhan komponen utama TFX:

- ExampleGen  
- StatisticsGen  
- SchemaGen  
- ExampleValidator  
- Transform  
- Trainer  
- Evaluator  
- Pusher  

Seluruh artefak pipeline tersimpan di direktori:

- `yrbror-pipeline/`

---

## 1. Dataset

Dataset yang digunakan adalah **Adult Income Dataset** (UCI ML Repository, versi Kaggle):  
https://www.kaggle.com/datasets/wenruliu/adult-income-dataset

Ringkasan dataset:

- **Jumlah data**: ± **48.843 baris**  
- **Jumlah fitur**: 14 fitur input + 1 label  
- **Fitur numerik (contoh)**:  
  `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`  
- **Fitur kategorikal (contoh)**:  
  `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`  
- **Kolom target**: `income`  
  - Kelas: `<=50K` dan `>50K`

Dataset berada pada:

- `yrbror-pipeline/data/adult.csv`

---

## 2. Arsitektur Pipeline

Struktur inti pipeline:

- `yrbror-pipeline/pipeline/preprocessing.py`  
  Fungsi `preprocessing_fn` untuk Transform.  
- `yrbror-pipeline/pipeline/model.py`  
  Definisi model Keras dan fungsi `run_fn` untuk Trainer.  
- `yrbror-pipeline/artifacts/`  
  Artefak dari ExampleGen, StatisticsGen, SchemaGen, Transform, Trainer, Evaluator, Pusher.  
- `yrbror-pipeline/metadata/`  
  Metadata pipeline.  
- `yrbror-pipeline/serving_model_dir/`  
  SavedModel hasil Pusher.

---

## 3. Performa Model

Model merupakan klasifikasi biner untuk memprediksi pendapatan (`>50K` atau tidak). Evaluasi dilakukan menggunakan **TensorFlow Model Analysis (TFMA)** dengan metrik:

- **Metrik utama**: `BinaryAccuracy`  
- **Threshold**: `0.75` (75%)

Hasil evaluasi menunjukkan:

- **Akurasi (BinaryAccuracy)**: **0.8616 (~86%)**

Nilai ini melampaui threshold, sehingga model memperoleh status **BLESSED** dari Evaluator. Model kemudian di-*push* ke:

- `yrbror-pipeline/serving_model_dir/`

Direktori ini berisi struktur SavedModel lengkap seperti `saved_model.pb`, `fingerprint.pb`, `keras_metadata.pb`, serta folder `assets/` dan `variables/` yang siap digunakan untuk deployment.

---

## 4. Opsi Deployment (Model Serving)

### 4.1. Endpoint Serving Model (Akses Publik)

Model di-serve menggunakan TensorFlow Serving pada port 8502 dan diekspos ke internet menggunakan ngrok.

Endpoint status model yang digunakan:

https://protogynous-unrousing-thresa.ngrok-free.dev/v1/models/adult_model


Endpoint tersebut mengembalikan `model_version_status` dengan `state: "AVAILABLE"` dan `error_code: "OK"`, yang menunjukkan bahwa model berhasil dideploy dan siap menerima request prediksi.


## 5. Monitoring Sistem dengan Prometheus

Sistem serving model dimonitor menggunakan **Prometheus**, yang mengumpulkan metrik dari service `tf_serving`. Prometheus menampilkan metrik dalam bentuk grafik time series.

Metrik yang berhasil dikumpulkan:

- **scrape_duration_seconds**  
  Durasi scraping Prometheus terhadap target:
instance="host.docker.internal:8501"
job="tf_serving"


Visualisasi monitoring terdapat pada:

- **yrbror-monitoring.png**

Grafik menunjukkan perubahan nilai `scrape_duration_seconds` dari waktu ke waktu, yang membuktikan bahwa:

- Prometheus berhasil terhubung ke service model.
- Scraping berjalan secara berkala.
- Time series metrik tercatat dengan benar.

Integrasi monitoring dapat dinyatakan **berhasil dan berjalan dengan baik**.

---

## 6. Cara Menjalankan Proyek Secara Lokal

### 6.1. Struktur Proyek

- `yrbror-pipeline/data/adult.csv`  
- `yrbror-pipeline/pipeline/preprocessing.py`  
- `yrbror-pipeline/pipeline/model.py`  
- `yrbror-pipeline/artifacts/`  
- `yrbror-pipeline/metadata/`  
- `yrbror-pipeline/serving_model_dir/`  

### 6.2. Setup Environment (Python 3.9 / WSL)

```bash
sudo apt update && sudo apt install -y python3.9 python3.9-venv
python3.9 -m venv ~/.venv-tfx39 && source ~/.venv-tfx39/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m ipykernel install --user --name=tfx-wsl-py39 --display-name "Python (tfx-wsl-py39)"