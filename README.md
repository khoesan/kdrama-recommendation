# Sistem Rekomendasi K-Drama dengan Content-Based Filtering

---

## 1. Project Overview

Drama Korea (K-Drama) telah menjadi fenomena global dengan jutaan penggemar dari berbagai negara. Dengan bertambahnya jumlah judul drama, penonton seringkali merasa kesulitan memilih tontonan yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang dapat membantu pengguna menemukan drama baru yang relevan berdasarkan selera pribadi mereka.

Proyek ini bertujuan membangun sistem rekomendasi drama Korea menggunakan pendekatan Content-Based Filtering yang memanfaatkan kemiripan konten, khususnya genre dan sinopsis drama. Dengan demikian, sistem dapat merekomendasikan drama yang memiliki karakteristik serupa dengan drama yang disukai pengguna sebelumnya.

Referensi yang mendukung pentingnya sistem rekomendasi ini termasuk studi terkait algoritma content-based filtering yang umum digunakan dalam berbagai platform media (Ricci et al., 2015) serta tren meningkatnya konsumsi K-Drama secara global (Statista, 2023).

---

## 2. Business Understanding

### Problem Statement
Banyak penonton K-Drama menghadapi kesulitan dalam menemukan drama baru yang sesuai dengan preferensi mereka akibat banyaknya pilihan yang tersedia.

### Goals
- Membangun sistem rekomendasi yang memberikan daftar drama Korea mirip berdasarkan genre dan sinopsis drama favorit pengguna.
- Menyediakan rekomendasi top-5 drama yang relevan dan berkualitas tinggi.

### Solution Approach
Dalam proyek ini, terdapat dua pendekatan solusi yang dipertimbangkan:

1. **Content-Based Filtering**  
   Sistem merekomendasikan drama berdasarkan kesamaan konten drama yang sudah disukai pengguna, khususnya menggunakan fitur teks (genre + sinopsis).

2. **Collaborative Filtering (Alternatif untuk Pengembangan Selanjutnya)**  
   Sistem merekomendasikan drama berdasarkan preferensi pengguna lain yang memiliki selera serupa. Pendekatan ini membutuhkan data interaksi pengguna yang tidak tersedia dalam dataset ini, sehingga menjadi solusi lanjutan.

---

## 3. Data Understanding

Dataset yang digunakan adalah **Top 250 Korean Dramas** yang tersedia di Kaggle:  
[https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset](https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset)

### Informasi Data

Dataset ini berisi 250 drama Korea dengan fitur sebagai berikut:

| Nama Variabel           | Deskripsi                                                                                              | Contoh                                      |
|------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------|
| **Name**               | Judul drama                                                                                           | Move to Heaven                              |
| **Aired Date**         | Tanggal tayang perdana                                                                                | 14-May-21                                   |
| **Year of release**    | Tahun rilis drama                                                                                      | 2021                                        |
| **Original Network**   | Jaringan televisi atau platform asli yang menayangkan drama                                          | Netflix                                     |
| **Aired On**           | Hari tayang drama                                                                                     | Friday                                      |
| **Number of Episodes** | Jumlah episode drama                                                                                   | 10                                          |
| **Duration**           | Durasi rata-rata per episode                                                                          | 52 min.                                     |
| **Content Rating**     | Rating konten, misalnya batas usia dan kategori konten                                               | 18+ Restricted (violence & profanity)       |
| **Rating**             | Rating skor drama (misalnya IMDb rating)                                                             | 9.2                                         |
| **Synopsis**           | Sinopsis atau ringkasan cerita drama                                                                 | Geu Roo is a young autistic man...          |
| **Genre**              | Genre drama (bisa lebih dari satu)                                                                   | Life, Drama, Family                         |
| **Tags**               | Tag tambahan yang menjelaskan tema dan elemen cerita                                                | Autism, Uncle-Nephew Relationship, Death... |
| **Director**           | Nama sutradara drama                                                                                  | Kim Sung Ho                                 |
| **Screenwriter**       | Nama penulis skenario drama                                                                           | Yoon Ji Ryun                                |
| **Cast**               | Daftar pemeran utama drama                                                                            | Lee Je Hoon, Tang Jun Sang, Hong Seung Hee...|
| **Production companies** | Perusahaan produksi yang memproduksi drama                                                          | Page One Film, Number Three Pictures        |
| **Rank**               | Peringkat drama berdasarkan rating atau popularitas                                                | #1                                          |

---

### Statistik dan Distribusi Data

- Total drama: 250 judul.
- Tahun rilis berkisar dari tahun 2000-an hingga 2021.
- Rating rata-rata: 7.5 – 9.5, dengan sebagian besar drama memiliki rating tinggi.
- Durasi rata-rata episode berkisar 50-60 menit.
- Genre umum: Drama, Romance, Life, Family, Fantasy, Thriller.
- Content rating bervariasi, termasuk kategori umum dan 18+ Restricted.

---

### Exploratory Data Analysis (EDA)

Visualisasi distribusi rating dan jumlah drama berdasarkan tahun rilis dapat membantu memahami karakteristik dataset.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribusi rating
plt.figure(figsize=(8,4))
sns.histplot(df['Rating'], bins=20, kde=True, color='salmon')
plt.title('Distribusi Rating Drama Korea')
plt.xlabel('Rating')
plt.ylabel('Jumlah Drama')
plt.show()

# Jumlah drama berdasarkan tahun rilis
plt.figure(figsize=(8,4))
sns.countplot(x='Year of release', data=df, palette='Blues_r')
plt.title('Jumlah Drama Berdasarkan Tahun Rilis')
plt.xticks(rotation=45)
plt.ylabel('Jumlah Drama')
plt.show()
````

---

## 4. Data Preparation

Untuk membuat sistem rekomendasi berbasis teks, dilakukan langkah-langkah berikut:

* Memilih fitur yang relevan: `Title`, `Synopsis`, `Genre`, dan `Rating`.
* Menghapus data duplikat berdasarkan `Title`.
* Mengatasi nilai kosong (null) di kolom `Synopsis` dan `Genre`.
* Menggabungkan kolom `Genre` dan `Synopsis` menjadi satu fitur teks (`combined_features`) yang akan digunakan dalam perhitungan kemiripan.

```python
df_filtered = df[['Title', 'Synopsis', 'Genre', 'Rating']].copy()
df_filtered.drop_duplicates(subset='Title', inplace=True)
df_filtered.dropna(subset=['Synopsis', 'Genre'], inplace=True)
df_filtered['combined_features'] = df_filtered['Genre'].fillna('') + " " + df_filtered['Synopsis'].fillna('')
```

**Alasan:**
Penggabungan genre dan sinopsis bertujuan untuk menangkap informasi baik dari kategori genre maupun isi cerita, sehingga sistem dapat mengenali kemiripan secara lebih menyeluruh.

---

## 5. Modeling and Results

### Pendekatan Content-Based Filtering

* Menggunakan **TF-IDF Vectorizer** untuk mengubah teks `combined_features` menjadi representasi numerik.
* Menghitung **cosine similarity** antar drama untuk mengukur kemiripan konten.
* Membuat fungsi rekomendasi `recommend_drama(title)` yang mengembalikan 5 drama paling mirip dengan input judul drama.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_filtered['combined_features'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df_filtered.index, index=df_filtered['Title']).drop_duplicates()

def recommend_drama(title, similarity=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan."
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    drama_indices = [i[0] for i in sim_scores]
    return df_filtered[['Title', 'Genre', 'Rating']].iloc[drama_indices]

# Contoh rekomendasi
recommend_drama("My Mister")
```

### Alternatif Pendekatan: CountVectorizer

Sebagai pembanding, menggunakan **CountVectorizer** untuk representasi teks yang lebih sederhana.

```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_filtered['combined_features'])
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)

def recommend_drama_count(title, similarity=cosine_sim_count):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan."
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    drama_indices = [i[0] for i in sim_scores]
    return df_filtered[['Title', 'Genre', 'Rating']].iloc[drama_indices]

# Contoh rekomendasi
recommend_drama_count("My Mister")
```

### Kelebihan dan Kekurangan

| Pendekatan      | Kelebihan                                                 | Kekurangan                                                              |
| --------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| TF-IDF          | Menangkap nuansa semantik, meminimalisasi bobot kata umum | Lebih kompleks, membutuhkan tuning parameter                            |
| CountVectorizer | Sederhana dan mudah dipahami                              | Bias terhadap kata yang sering muncul, kurang sensitif terhadap konteks |

---

## 6. Evaluation

### Metrik Evaluasi

Karena proyek ini menggunakan pendekatan content-based tanpa data interaksi pengguna, evaluasi dilakukan dengan:

* **Skor cosine similarity rata-rata** dari top-N rekomendasi untuk tiap judul drama.
* **Evaluasi kualitatif** dengan melihat kesamaan genre dan tema drama yang direkomendasikan.

### Formula Skor Similarity Rata-Rata

{Avg Similarity} = \frac{1}{N} \sum_{i=1}^{N} \text{cosine\_similarity}(d_{input}, d_i)

di mana $d_{input}$ adalah drama yang menjadi input dan $d_i$ adalah drama hasil rekomendasi.

```python
def evaluate_similarity(title, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan."
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    avg_score = sum([score for _, score in sim_scores]) / top_n
    return f"Rata-rata skor similarity dari top-{top_n} rekomendasi untuk '{title}': {avg_score:.4f}"

evaluate_similarity("My Mister")
```

### Hasil dan Insight

Misalnya untuk input "My Mister", sistem menghasilkan rekomendasi drama dengan genre dan rating yang mirip, yang menunjukkan konsistensi dalam menghasilkan rekomendasi relevan secara semantik.

---

*Laporan ini disusun untuk memenuhi persyaratan submission pada platform Dicoding.*
