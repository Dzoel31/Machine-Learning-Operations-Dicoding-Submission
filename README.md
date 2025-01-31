# Submission 1: Sentence Sentiment Analysis

Nama: Dzulfikri Adjmal

Username dicoding: dzulfikriadjmal

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Sentiment and Emotion Analysis Dataset](<https://www.kaggle.com/datasets/kushagra3204/sentiment-and-emotion-analysis-dataset>) (Hanya menggunakan data combined_sentimen) |
| Masalah | Pada era digital saat ini, banyak sekali data yang dihasilkan oleh pengguna media sosial. Data ini dapat digunakan untuk menganalisis sentimen dari pengguna terhadap suatu topik tertentu. Setiap topik dapat memiliki sentimen baik positif maupun negatif. Oleh karena itu, diperlukan model machine learning yang dapat mengklasifikasikan sentimen dari suatu kalimat. Dengan adanya model ini, kita dapat melakukan filter terhadap komentar atau tulisan yang ada pada suatu platform dan menentukan langkah yang tepat untuk mengatasi sentimen negatif yang muncul. |
| Solusi machine learning | Solusi yang diusulkan adalah dengan menggunakan model machine learning yang dapat mengklasifikasikan sentimen dari suatu kalimat. Model yang digunakan merupakan model klasifikasi biner yang dapat memprediksi sentimen positif atau negatif dari suatu kalimat. |
| Metode pengolahan | Pengolahan data yang digunakan meliputi penghapusan karakter khusus dan label encoding pada kolom `sentiment`. |
| Arsitektur model | Arsitektur model yang digunakan merupakan model klasifikasi biner sederhana yang terdiri dari `TextVectorization`, `Embedding`, `GlobalAveragePooling1D`, dan `Dense`. Aktivasi yang digunakan pada layer adalah `relu` untuk mempercepat konvergensi model dan `sigmoid` pada layer output untuk menghasilkan output dua kelas dalam bentuk probabilitas. |
| Metrik evaluasi | Metrik evaluasi yang digunakan berupa AUC dan BinaryClassfier. AUC merupakan metrik evaluasi yang digunakan untuk mengevaluasi performa model dalam memprediksi label, sedangkan BinaryClassifier digunakan untuk melihat seberapa baik model dalam memprediksi label positif |
| Performa model | Pada proses training, model menghasilkan `binary_accuracy` sebesar 0.9405 dengan `val_binary_accuracy` sebesar 0.7080. Nilai ini menunjukkan terdapat overfitting pada model. Perlu dilakukan peningkatan performa model dengan melakukan tuning hyperparameter atau menggunakan model yang lebih kompleks. |
| Opsi deployment | Deployment model yang diusulkan adalah dengan menggunakan model serving yang di deploy pada cloud Railway. |
| Web app | [Sentiment Sentence Classification](https://dzulfikri-sentiment-sentence.up.railway.app/v1/models/sentence-sentiment/metadata)|
| Monitoring | Penggunaan prometheus dan grafana memudahkan untuk memantau kinerja model yang telah di deploy terhadap beberapa metrik. Pada prometheus dapat dilihat metrik yang dipantau adalah jumlah request dan pada Grafana ditambahkan panel untuk memantau latensi dari model. Keuntungan dari monitoring ini adalah memudahkan untuk mengetahui kinerja model dan memperbaiki jika terjadi masalah, serta meningkatkan performa model. |
