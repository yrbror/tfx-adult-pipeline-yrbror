# Submission 1: TFX Adult Income Classification
Nama: Muhammad Faiz Abror  
Username dicoding: yrbror  

| | Deskripsi |
| ----------- | ----------- |
| **Dataset** | Adult Income Dataset (UCI ML Repository — Kaggle) https://www.kaggle.com/datasets/wenruliu/adult-income-dataset |
| **Masalah** | Memprediksi apakah seorang individu memiliki pendapatan tahunan >50K berdasarkan fitur demografis. |
| **Solusi machine learning** | Klasifikasi biner menggunakan pipeline machine learning berbasis **TensorFlow Extended (TFX)** dengan preprocessing otomatis dan model Keras. |
| **Metode pengolahan** | Statistik data (StatisticsGen), deteksi anomali (ExampleValidator), skema otomatis (SchemaGen), transformasi fitur (Transform), dan pelatihan model (Trainer). |
| **Arsitektur model** | Model **Keras Sequential** yang dilatih dalam pipeline TFX. Output model disimpan dalam format SavedModel melalui komponen Pusher. |
| **Metrik evaluasi** | `BinaryAccuracy` dengan threshold **0.75** menggunakan **TensorFlow Model Analysis (TFMA)**. |
| **Performa model** | Akurasi evaluasi (**BinaryAccuracy**) = **0.8616 (~86%)**, model berstatus **BLESSED**. |
| **Opsi deployment** | Model di-deploy menggunakan **TensorFlow Serving (Docker)** dan diekspos ke cloud menggunakan **ngrok** (endpoint sementara). Bukti deployment diberikan melalui screenshot. |
| **Web app** | Karena endpoint ngrok bersifat sementara, bukti akses disajikan melalui screenshot: **yrbror-deployment.png** |
| **Monitoring** | Monitoring dilakukan menggunakan **Prometheus** dengan metrik `scrape_duration_seconds`. Bukti monitoring disertakan pada **yrbror-monitoring.png**. |