# Laporan Proyek Machine Learning - Aulia Anshari Fathurrahman

## Domain Proyek

Dalam proyek ini, saya akan menganalisis kumpulan data pemasaran bank dan membuat algoritma klasifikasi menggunakan feature engineering dan EDA. Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai keuangan dan improvised market campaigning dengan judul proyek "Bank Campaign Predictive Analytics".

### Latar Belakang

Pendapatan bank-bank Portugis menurun dan mereka ingin tahu apa yang harus dilakukan. Penelitian menunjukkan bahwa akar masalahnya adalah nasabah tidak melakukan setoran sesering biasanya. Deposito berjangka memungkinkan bank untuk menyimpan deposito untuk jangka waktu tertentu, memungkinkan bank untuk berinvestasi dalam instrumen keuangan hasil tinggi dan mendapatkan keuntungan. Selain itu, bank lebih cenderung membujuk nasabah deposito berjangka mereka untuk membeli produk lain seperti reksa dana dan asuransi untuk lebih meningkatkan pendapatan mereka. Untuk alasan ini, sebuah bank Portugis ingin mengidentifikasi nasabah lama yang kemungkinan akan menerima deposito berjangka dan memfokuskan upaya pemasarannya pada nasabah tersebut.

Berdasarkan permasalahan diatas, pada proyek ini akan dibangun model machine learning untuk memprediksi nasabah lama yang kemungkinan akan menerima deposito berjangka. Dengan adanya model machine learning ini, di harapkan dapat membantu dan memudahkan analisa serta dapat mengambil keputusan yang tepat terkait pengidentifikasian nasabah dan dapat strategi marketing yang akan dilakukan bank. Kemudian untuk tahap pengembangan selanjutnya, di harapkan implementasi dari model ini dapat dijalankan pada sebuah aplikasi berbasis web ataupun android nantinya.

Referensi : [A data-driven approach to predict the success of bank telemarketing](https://www.sciencedirect.com/science/article/pii/S016792361400061X)

## Business Understanding

### Problem Statements

Berdasarkan pada latar belakang di atas, permasalahan yang dapat diselesaikan pada proyek ini adalah sebagai berikut:
- Bagaimana cara menganalisa data nasabah yang akan melakukan deposito berjangka dan melihat faktor yang memengaruhinya?  
- Bagaimana cara melakukan pre-processing dan feature engineering pada data tersebut? 
- Bagaimana cara membuat model machine learning untuk mengklasifikasi nasabah?

### Goals

Berdasarkan pernyataan masalah di atas, tujuan dibuatnya proyek ini adalah :
- Melakukan Exploratory Data Analysis pada data nasabah.
- Melakukan pre-processing data dan feature engineering agar dapat menghasilkan inputan model yang baik. 
- Melakukan pembuatan model klasifikasi nasabah yang akan melakukan deposito berjangka.

### Solution statements

Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini di antaranya:
- Melakukan Exploratory Data Analysis dengan menampilkan chart-chart terkait data, agar lebih mudah memahami data tersebut.
- Melakukan pre-processing data : Handling missing value dan outlier.
- Melakukan feature engineering : Encoding data, Standardization data, Feature selection dan Train tes split.
- Melakukan pembangunan model menggunakan 5 model classifiers sklearn : Logistic Regression, Random Forest, KNN, Decision Tree dan SGD
- menggunakan Pipeline model dan GridSearch agar pelatihan dapat berjalan lebih efektif
- menggunakan metric evaluasi ROC

## Data Understanding

Data ini menyangkut kampanye pemasaran langsung oleh lembaga perbankan Portugis. Kampanye pemasaran dilakukan melalui telepon. Beberapa kontak dengan pelanggan yang sama sering kali diperlukan untuk mengonfirmasi apakah suatu produk (deposito berjangka bank) berlangganan (“ya”) atau tidak (“tidak”). 

Data ini berjumlah 41188 baris dan 21 features, diurutkan berdasarkan tanggal (dari Mei 2008 - November 2010).

Sumber: [Bank Marketing Campaign](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

### Variabel-variabel pada Bank Marketing dataset adalah sebagai berikut:

#### Attributes: Bank client data:

- Age : Age of the lead (numeric)
- Job : type of job (Categorical)
- Marital : Marital status (Categorical)
- Education : Educational Qualification of the lead (Categorical)
- Default: Does the lead has any default(unpaid)credit (Categorical)
- Housing: Does the lead has any housing loan? (Categorical)
- loan: Does the lead has any personal loan? (Categorical)

#### Related with the last contact of the current campaign:

- Contact : Contact communication type (Categorical)
- Month : last contact month of year (Categorical)
- day_of_week : last contact day of the week (categorical)
- duration : last contact duration, in seconds (numeric). Important note: Duration highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### Other attributes:

- campaign : number of contacts performed during this campaign and for this client (numeric)
- pdays : number of days that passed by after the client was last contacted from a previous campaign(numeric; 999 means client was not previously contacted))
- previous : number of contacts performed before this campaign and for this client (numeric)
- poutcome : outcome of the previous marketing campaign (categorical)

#### Social and economic context attributes

- emp.var.rate : employment variation rate - quarterly indicator (numeric)
- cons.price.idx : consumer price index - monthly indicator (numeric)
- cons.conf.idx : consumer confidence index - monthly indicator (numeric)
- euribor3m : euribor 3 month rate - daily indicator (numeric)
- nr.employed : number of employees - quarterly indicator (numeric)

#### Output variable (desired target):

- y : has the client subscribed a term deposit? (binary: 'yes','no')

#### Data info:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/info.png?raw=true)

### Exploratory Data Analysis

- Cek distribusi data categorical terhadap y:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/distofjobanddeposit.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/distofmaritalanddeposit.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/distofeducationanddeposit.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/distofcontactanddeposit.png?raw=true)

- Cek distribusi numerical data terhadap y:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/DurationCallofCampaign.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/DurationCallofJob.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/histofemp.var.rate.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/histofcons.price.idx.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/histofcons.conf.idx.png?raw=true)

- Cek korelasi data:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/corr.png?raw=true)

## Data Preparation

Berikut tahapan data preparation yang dilakukan:

### Handle data noise

- Menghitung data null, dengan menjumlahkan null value menggunakan function isnull pandas. (hasil : tidak ditemukan data null)
- Menghilangkan data dengan value 'unknown' pada semua kolom, dengan menggunakan function drop pandas. (hasil : shape data menjadi 30000 rows)
- value 'unknown' dihilangkan karena value tersebut memberikan hasil yang bias kepada data target

### Handle outlier data

- Cek outlier data menggunakan boxplot

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofageb.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofdurationb.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofcampaignb.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofcons.price.idx.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofcons.conf.idx.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofeuribor3m.png?raw=true)

- Ditemukan 3 data yang memiliki outlier yaitu, ['age', 'duration', 'campaign']
- Outlier dihilangkan dengan memfilter data pada kuartil pertama dikurangkan dengan 1.5 lalu dikalikan IQR, dan pada kuartil ketiga ditambahkan 1.5 lalu dikalikan IQR
- filter = (data[cols] >= Q1 - 1.5 * IQR) & (data[cols] <= Q3 + 1.5 *IQR)
- hasil :

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofage.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofduration.png?raw=true)

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/boxplotofcampaign.png?raw=true)

### Feature Engineering

#### Frequency Encoding

- Dilakukan grouping pada value [education] terlebih dahulu yang memiliki value 'basic.*y' menjadi satu value 'middle school'. Ini dilakukan agar value pada [education] tidak terlalu skewness
- frequency encoding dilakukan pada 2 variabel, yaitu [education] dan [job]
-- dengan menghitung jumlah dari masing-masing value dan dibagi dengan banyak data (n[i] = total[n][i]/total[data])
-- setelah itu dilakukan mapping dan dibuatkan kolom baru pada dataset, lalu menghapus kolom lama

#### Encoding month and day

- Manual encoding menggunakan dictionary dan mapping

#### Encoding [pdays] value

- pada [pdays] terdapat value '999', berdasarkan informasi dari dataset, value ini memberikan informasi bahwa client ini belum pernah dihubungi sebelumnya, berbeda dengan value lain yang memaksudkan hari yang dilewati setelah client tersebut dihubungi
- jadi pada value ini kita encoding menjadi '0' dengan menggunakan lambda

#### Label Encoding

- label encoding dilakukan menggunakan LabelEncoder sklearn.
- data yang di encoding adalah ['housing', 'default', 'loan', 'y']
-- value kolom ini berisikan yes dan no
-- maka value tersebut akan diubah menjadi 0 dan 1

#### One-hot Encoding

- one-hot encoding menggunakan get_dummies pandas
- data yang di encoding adalah ['poutcome', 'contact']
-- value kolom ini sebelumnya berupa kategori
-- jadi value tersebut akan dikeluarkan menjadi kolom baru dan berisikan identitasnya
-- setelah itu kolom lama dihapus 

#### Target Encoding

- target encoding dilakukan menggunakan TargetEncoder sklearn
- data yang di encoding adalah ['marital']
-- target encoding mengubah value data sebelumnya dengan menghitung rata-rata value terhadap data target
-- setelah itu kolom lama dihapus

### Standardization numerical variable

- normalisasi data menggunakan StandarScaller sklearn
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

Pada proyek ini, Proses modeling dalam proyek ini menggunakan 5 algoritma machine learning yaitu Logistic Regression, Random Forest, KNN, Decision Tree dan SGD, kemudian membandingkan performanya.

### Build pipeline of classifiers

Disini kita menggunakan pipeline agar training model dapat berjalan dengan efektif.
Definisikan model-model yang telah diimport kedalam masing-masing pipeline, sehingga menjadi seperti:

LogisticRegression : 
pipe_lr = Pipeline([('lr', LogisticRegression(random_state=random_state, n_jobs=n_jobs, max_iter=500))])

RandomForestClassifier : 
pipe_rf = Pipeline([('rf', RandomForestClassifier(random_state=random_state, oob_score=True, n_jobs=n_jobs))])

KNeighborsClassifier : 
pipe_knn = Pipeline([('knn', KNeighborsClassifier(n_jobs=n_jobs))])

DecisionTreeClassifier : 
pipe_dt = Pipeline([('dt', DecisionTreeClassifier(random_state=random_state, max_features='auto'))])

SGDClassifier : 
pipe_sgd = Pipeline([('sgd', SGDClassifier(random_state=random_state, n_jobs=n_jobs, max_iter=1500))])

### Set parameters for Grid Search

Kita melakukan hyperparameter tuning menggunakan Grid Search.
Grid Search merupakan sebuah proses yang mencari secara mendalam melalui subset yang ditentukan secara manual dari ruang hyperparameter dari algoritma yang ditargetkan.
Penentuan hyperparameter yang kita berikan sesuai dengan ketentuan inputan parameters yang diterima model. Lihat sklearn documentations terkait model.

### Grid Search Object

Buatkan object yang akan kita panggil saat memulai training. Pembuatan object menggunakan GridSearchCV dengan input pipeline model, hyperparameter dan metrics. 
-- GridSearchCV(pipe_{model}, param_grid=grid_params_{model},scoring='accuracy', cv=cv) 

### Model Training

Sebelum dilakukan training, buat dict kosong untuk tempat meletakkan hasil metrics training.
Training dilakukan dengan memanggil object grid Search. Training data akan dilakukan sesuai hyperparameter yang telah di set, dan akan mengeluarkan best_score untuk disimpan ke dalam dict berisikan metrics.

## Evaluation

Pada proyek ini, metrik evaluasi yang digunakan untuk mengukur kinerja model yaitu menggunakan metrik akurasi dan ROC. Akurasi di sini merupakan tingkat keakuratan data prediksi yang didasarkan dari data latih pada model, ROC sendiri merupakan Receiver Operating Characteristic, semacam alat ukur performance untuk classification problem dalam menentukan threshold dari suatu model. Kita menggunakan Kurva ROC untuk menganalisis kekuatan prediksi pengklasifikasi: kurva tersebut menyediakan cara visual untuk mengamati bagaimana perubahan treshold klasifikasi model memengaruhi kinerja model. Kurva memungkinkan kita untuk memilih treshold klasifikasi yang memungkinkan model kita untuk mengidentifikasi sebanyak mungkin positif sejati sambil meminimalkan positif palsu.

### ROC

Kurva ROC dibuat berdasarkan nilai telah didapatkan pada perhitungan dengan confusion matrix, yaitu antara False Positive Rate dengan True Positive Rate. Dimana:
-- False Positive Rate (FPR) = False Positive / (False Positive + True Negative)
-- True Positive Rate (TPR) = True Positive / (True Positive + False Negative)

Untuk membaca kurva ROC sangat mudah, kinerja algoritma klasifikasi adalah:
-- JELEK, jika kurva yang dihasilkan mendekati garis baseline atau garis yang melintang dari titik 0,0.
-- BAGUS, jika kurva mendekati titik 0,1.

####  ROC AUC

Area Under Curve (AUC) membuat kita mudah dalam membandingkan model satu dengan yang lainnya. AUC adalah luas area di bawah curve ROC, atau integral dari fungsi ROC. semakin tinggi nilai AUC, maka semakin bagus pula hasil model kita

### Model result

Berikut plot hasil training dari model :

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/modelresult.png?raw=true)

Berikut tabel hasil metrics:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/tabelhasil.png?raw=true)

#### Dapat kita ambil kesimpulan bahwa : model terbaik yaitu menggunakan RandomForestClassifier dengan accuracy [0.923578] dan roc_auc [0.935259] 

Berikut plot ROC Curve dari RandomForestClassifier terhadap data kita:

![alt text](https://github.com/auliaanshari/Bank-marketing-Predictive-analytics/blob/main/image/ROCcurve.png?raw=true)
