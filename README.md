# Laporan Proyek Machine Learning - Aulia Anshari Fathurrahman

## Project Overview

Dalam proyek ini, penulis akan menganalisis kumpulan data pariwisata di beberapa kota di Indonesia dan membuat algoritma sistem rekomendasi. Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai pariwisata dan sistem rekomendasi dengan judul proyek "Indonesia Tourism Recommendation System".

### Latar Belakang

Sebelum melakukan perjalanan, seseorang biasanya membuat rencana terlebih dahulu tentang tempat yang akan dikunjungi dan kapan harus berangkat. Hal ini dilakukan untuk menghindari masalah. Salah satunya adalah jarak dan durasi perjalanan yang berbeda dari yang diharapkan.

Indonesia memiliki tempat-tempat wisata yang menarik seperti pedalaman yang indah, situs budaya dan sejarah yang menarik, pantai dan lain-lain. Namun karena Covid-19 baru-baru ini jumlah wisatawan menurun. Saat ini banyak wisata yang kembali beroperasi di beberapa kota di Indonesia, dan ini kembali menjadi momentum yang tepat untuk mendongkrak pariwisata.

Berdasarkan permasalahan diatas, pada proyek ini akan dibangun model machine learning untuk merekomendasikan pariwisata berdasarkan rating pengguna dan konten pariwisata. Dengan adanya model machine learning ini, diharapkan dapat membantu dan menunjang sektor pariwisata Indonesia. Kemudian untuk tahap pengembangan selanjutnya, di harapkan implementasi dari model ini dapat dijalankan pada sebuah aplikasi berbasis web ataupun android nantinya.

## Business Understanding

### Problem Statements

Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:
- Bagaimana cara preprocessing dan menganalisa feature-feature data pariwisata yang diperlukan?  
- Bagaimana cara membuat model yang merekomendasi object berdasarkan konten dan preferensi pengguna?

### Goals

Berdasarkan pernyataan masalah di atas, tujuan dibuatnya proyek ini adalah :
- Melakukan pre-processing data Exploratory Data Analysis pada data pariwisata.
- Melakukan pembuatan model sistem rekomendasi berdasarkan konten dan preferensi pengguna.

### Solution statements

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:
- Melakukan Exploratory Data Analysis dengan menampilkan info dan chart terkait data, agar lebih mudah memahami data tersebut.
- Melakukan pre-processing data : Handling missing value dan outlier.
- Melakukan feature engineering : Encoding data, Normalisasi data, Feature selection dan Train tes split.
- Melakukan pembangunan model Content Based Filter menggunakan TF-IDF Vectorizer dan Cosine Similarity 
- Melakukan pembangunan model Colaborative Filtering menggunakan RecommenderNet
- Menampilkan hasil training dengan TensorBoard
- Implementasi model dengan menampilkan top 5 menggunakan kedua metode

## Data Understanding

Data ini menyangkut data pariwisata, data user dan rating yang diberikan pengguna. Pengguna memberikan rating untuk tempat yang telah dikunjunginnya. Rating yang diberikan pengguna dalam skala 1 - 5. 

Data ini terdiri 3 file berbeda untuk setiap konteksnya.

### Variabel-variabel pada Indonesia Tourism Destination dataset adalah sebagai berikut:

#### Tourism info:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/infotourism.png?raw=true)

#### Attributes of tourism data:

- Place_Id : ID of tourism (numeric)
- Place_Name : Name of tourism (Categorical)
- Description : Description of tourism (Categorical)
- Category : Category of tourism (Categorical)
- City : City where tourism exists (Categorical)
- Price : ticket Price (numeric)
- Rating: rating of tourism (numeric)
- Time_Minutes: minutes of ? (numeric)
- Coordinate: coordinate of tourism (object)
- Lat: latitude of tourism (numeric)
- Long: longitude of tourism (numeric)
- Unnamed 11: ?(numeric)
- Unnamed 12: ? (numeric)

contains (437) data and 13 features

#### User info:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/infouser.png?raw=true)

#### Attributes of user data:

- User_Id : ID of user (numeric)
- Location : city where user from (Categorical)
- Age : age of user (numeric)

contains (300) data and 3 features

#### User info:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/inforating.png?raw=true)

#### Attributes of rating data:

- User_Id : ID of user (numeric)
- Place_Id : ID of tourism(numeric)
- Place_Ratings : Ratings that user gives (categorical)

contains (10000) data and 3 features

Sumber: [Kaggle](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination)

### Exploratory Data Analysis

- Cek distribusi data categorical:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/distofCategory.png?raw=true)

Dapat dilihat kategori budaya, taman hiburan dan cagar alam menjadi data yang cukup dominan dan tempat ibadah menjadi data yang paling sedikit pada dataset.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/distofCity.png?raw=true)

Dapat dilihat kota Jakarta dan Yogyakarta menjadi data dengan tourism yang paling banyak pada dataset.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/AgeofUser.png?raw=true)

Dapat dilihat distribusi umur dari user cukup beragam dengan user paling banyak berusia 30.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/Top10LocofUser.png?raw=true)

Dapat dilihat top 10 user berdasarkan lokasi, paling banyak dari Bekasi, Jawa Barat.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/Top10Rating.png?raw=true)

Dapat dilihat top 10 tourism berdasarkan rating, paling banyak pada Gunung Lalakon dan Pantai Parangtritis.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/RatingUserGive.png?raw=true)

Dapat dilihat rating yang diberikan user cukup seimbang.

- Cek distribusi numerical data:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/boxplotprice.png?raw=true)

Dapat dilihat boxplot price terfokus kepada 0. ini kemungkinan harga ticket tourism kebanyak free atau tidak perlu ticket.

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/boxplotrating.png?raw=true)

Dapat dilihat boxplot rating cukup seimbang.

## Data Preparation

Berikut tahapan data preparation yang dilakukan:

### Handle missing value

- Menghitung data null pada turism, dengan menjumlahkan null value menggunakan function isnull pandas. (hasil : ditemukan data null pada kolom time minutes)
- drop kolom yang tidak diperlukan, dengan menggunakan function drop pandas. (hasil : drop Time minutes, unnamed 11 dan 12)

### Handle outlier data

- Cek outlier data menggunakan boxplot

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/boxplotprice.png?raw=true)

We saw the boxplot dist centered to 0,  
it because some value price of tourism is 0,  
assume the 0 value is free ticket, so we dont see this as outliers.

### normalisasi numerical variable

- normalisasi data menggunakan MinMaxScaller sklearn
- feature yang dipilih adalah yang berupa numerical feature
- setelah itu data disimpan ke variabel baru

### Train Test Split

- split data menggunakan train_test_split sklearn
- ukuran split data, yaitu 80 % data train, 20 % data test
- setelah split data, variabel X_train, X_test, y_train, y_test akan dikeluarkan
-- X_train: (21041, 21)
-- X_test: (5261, 21)
-- y_train: (21041,)
-- y_test: (5261,)

## Modelling

Pada proyek ini, Proses modeling dalam proyek ini menggunakan 2 metode sistem rekomendasi yaitu Content Based Filter dan Collaborative Filtering.

### Content Based Filter

Pada modeling Content Based Filtering, langkah pertama yang dilakukan ialah penulis menggunakan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap kategori dan kota tourism. Fungsi yang penulis gunakan adalah tfidfvectorizer() dari library sklearn. Selanjutnya penulis melakukan fit dan transformasi ke dalam bentuk matriks. Keluarannya adalah matriks berukuran (437, 15). Nilai 437 merupakan ukuran data tourism dan 15 merupakan matriks kategori dan kota.

Untuk menghitung derajat kesamaan (similarity degree) antar tourism, penulis menggunakan teknik cosine similarity dengan fungsi cosine_similarity dari library sklearn. Berikut dibawah ini adalah rumusnya:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/cosine.png?raw=true)

Langkah selanjutnya yaitu menggunakan argpartition untuk mengambil sejumlah nilai k tertinggi dari similarity data kemudian mengambil data dari bobot (tingkat kesamaan) tertinggi ke terendah. Kemudian menguji akurasi dari sistem rekomendasi ini untuk menemukan rekomendasi tourism yang mirip dari tourism yang ingin dicari.

#### Kelebihan

Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
#### Kekurangan

Hanya dapat digunakan untuk fitur yang sesuai, seperti f, dan buku.
Tidak mampu menentukan profil dari user baru.

Berikut ini adalah konten yang dijadikan referensi untuk menentukan 5 rekomendasi wisata yang memiliki kesamaan feature yang sama:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/cbfcari.png?raw=true)

Terlihat pada tabel diatas bahwasannya saya akan menguji coba model berdasarkan 'Farm House Susu Lembang' dengan kategori Taman Hiburan dan kota Bandung.

Berikut ini adalah hasil rekomendasi tertinggi dari model Content Based Filtering berdasarkan referensi tourism diatas:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/hasilcbf.png?raw=true)


### Collaborative Filtering

Pada modeling Collaborative Filtering penulis menggunakan data hasil gabungan dari dua datasets yaitu torusim & rating. Langkah pertama adalah melakukan encode data User_Id & Place_Id setelah di encode lakukan mapping ke dalam data yang digunakan dan juga mengubah nilai rating menjadi float. Selanjutnya ialah membagi data untuk training sebesar 80% dan validasi sebesar 20%.

Lakukan proses embedding terhadap data tourism dan pengguna. Lalu lakukan operasi perkalian dot product antara embedding pengguna dan tourism. Selain itu, penulis juga menambahkan bias untuk setiap pengguna dan tourism. Skor kecocokan ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid. Untuk mendapatkan rekomendasi tourism, penulis mengambil sampel user secara acak dan mendefinisikan variabel not_visited yang merupakan daftar tourism yang belum pernah dikunjungi oleh pengguna.

#### Kelebihan

Tidak memerlukan atribut untuk setiap itemnya.
Dapat membuat rekomendasi tanpa harus selalu menggunakan dataset yang lengkap.
Unggul dari segi kecepatan dan skalabilitas.
Rekomendasi tetap akan berkerja dalam keadaan dimana konten sulit dianalisi sekalipun

####Kekurangan

Membutuhkan parameter rating, sehingga jika ada item baru sistem tidak akan merekomendasikan item tersebut.

Berikut ini adalah hasil rekomendasi tourism tertinggi terhadap user 121:

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/hasilcf.png?raw=true)

## Evaluation

Evaluasi yang akan penulis lakukan disini yaitu evaluasi dengan Root Mean Squared Error (RMSE) dan memplotnya ke tensorBoard

### TensorBoard

![alt text](https://github.com/auliaanshari/Indonesia-tourism-Recommendation-system/blob/main/image/tensorboard.png?raw=true)

# Referensi
[GetLoc](https://github.com/AgungP88/getloc-apps)
