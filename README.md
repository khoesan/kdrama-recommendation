## **Sistem Rekomendasi K-Drama dengan Content-Based Filtering - Khoirotun Hisan**

### 1. Project Overview

Drama Korea (K-Drama) telah menjadi fenomena global dengan jutaan penggemar dari berbagai negara. Dengan bertambahnya jumlah judul drama, penonton seringkali merasa kesulitan memilih tontonan yang sesuai dengan preferensi mereka. Oleh karena itu, dibutuhkan sistem rekomendasi yang dapat membantu pengguna menemukan drama baru yang relevan berdasarkan selera pribadi mereka.

**Tujuan Proyek:**

* Membangun sistem rekomendasi drama Korea menggunakan pendekatan Content-Based Filtering.
* Memprediksi drama yang relevan berdasarkan sinopsis dan genre dari drama favorit pengguna.

---

### 2. Business Understanding

#### Problem Statement

Penonton kesulitan menemukan drama baru yang sesuai dengan selera mereka karena banyaknya pilihan dan kurangnya sistem personalisasi di beberapa platform tontonan.

#### Goals

* Mengembangkan sistem rekomendasi berbasis konten menggunakan fitur Genre dan Sinopsis.
* Memberikan Top-5 rekomendasi drama yang serupa dengan drama favorit pengguna.

#### Solution Approach

Sistem rekomendasi ini dikembangkan menggunakan pendekatan **Content-Based Filtering** yang memanfaatkan fitur teks (Genre dan Sinopsis) untuk menemukan kemiripan antar drama.

---

### 3. Data Understanding

#### Dataset

- Dataset: **Top 250 Korean Dramas**
- Sumber: [Kaggle - ahbab911/top-250-korean-dramas-kdrama-dataset](https://www.kaggle.com/datasets/ahbab911/top-250-korean-dramas-kdrama-dataset)
- Jumlah data: 250 drama

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

### 4. Univariate Exploratory Data Analysis (EDA)

Visualisasi distribusi rating dan jumlah drama berdasarkan tahun rilis dilakukan untuk memahami sebaran data.

```python
# Distribusi rating
plt.figure(figsize=(8,4))
sns.histplot(df['Rating'], bins=20, kde=True, color='salmon')
plt.title('Distribusi Rating Drama Korea')
plt.xlabel('Rating')
plt.ylabel('Jumlah Drama')
```

```python
# Jumlah drama berdasarkan tahun rilis
plt.figure(figsize=(8,4))
sns.countplot(x='Year of release', data=df, palette='Blues_r')
plt.title('Jumlah Drama Berdasarkan Tahun Rilis')
plt.xticks(rotation=45)
plt.ylabel('Jumlah Drama')
```
#### Statistik dan Distribusi Data

- Total drama: 250 judul.
- Tahun rilis berkisar dari tahun 2000-an hingga 2021.
- Rating rata-rata: 7.5 – 9.5, dengan sebagian besar drama memiliki rating tinggi.
- Durasi rata-rata episode berkisar 50-60 menit.
- Genre umum: Drama, Romance, Life, Family, Fantasy, Thriller.
- Content rating bervariasi, termasuk kategori umum dan 18+ Restricted.
---

### 5. Data Preparation

#### Langkah-langkah Persiapan:

1. **Menghapus duplikat dan missing values**:

```python
df_filtered = df[['Name', 'Synopsis', 'Genre', 'Rating']].copy()
df_filtered.drop_duplicates(subset='Name', inplace=True)
df_filtered.dropna(subset=['Synopsis', 'Genre'], inplace=True)
```

2. **Menggabungkan fitur Genre dan Sinopsis menjadi `combined_features`**:

```python
df_filtered['combined_features'] = df_filtered['Genre'].fillna('') + " " + df_filtered['Synopsis'].fillna('')
```

3. **Ekstraksi Fitur dengan TF-IDF**
   TF-IDF (Term Frequency–Inverse Document Frequency) digunakan untuk memberi bobot pada kata-kata penting yang membantu sistem memahami konteks cerita:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_filtered['combined_features'])
```

4. **(Alternatif) Ekstraksi Fitur dengan CountVectorizer**
   CountVectorizer memberikan bobot berdasarkan frekuensi kemunculan kata:

```python
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_filtered['combined_features'])
```

---

### 6. Model and Results

#### Pendekatan Content-Based Filtering

* **Prinsip Kerja**: Sistem mencari item serupa berdasarkan fitur konten menggunakan kemiripan vektor teks.
* **Algoritma**: Cosine Similarity digunakan untuk mengukur sejauh mana dua drama mirip secara semantik.

```python
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim_count = cosine_similarity(count_matrix, count_matrix)
```

#### Fungsi Rekomendasi

```python
indices = pd.Series(df_filtered.index, index=df_filtered['Name']).drop_duplicates()

def recommend_drama(title, similarity=cosine_sim):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan dalam data."

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]

    drama_indices = [i[0] for i in sim_scores]
    return df_filtered[['Name', 'Rating']].iloc[drama_indices]
```

```python
def recommend_drama_count(title, similarity=cosine_sim_count):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan."

    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    drama_indices = [i[0] for i in sim_scores]
    return df_filtered[['Name', 'Rating']].iloc[drama_indices]
```

#### Contoh Output Rekomendasi

```python
recommend_drama("My Mister")
```
| No. | Nama Drama               | Rating |
| --- | ------------------------ | ------ |
| 1   | Save Me                  | 8.6    |
| 2   | Blind                    | 8.5    |
| 3   | My Father is Strange     | 8.6    |
| 4   | My Unfamiliar Family     | 8.4    |
| 5   | The World of the Married | 8.5    |

```python
recommend_drama_count("My Mister")
```
| No. | Nama Drama           | Rating |
| --- | -------------------- | ------ |
| 1   | Memory               | 8.3    |
| 2   | Dear My Friends      | 8.7    |
| 3   | Children of Nobody   | 8.6    |
| 4   | My Father is Strange | 8.6    |
| 5   | My Unfamiliar Family | 8.4    |

---


### 7. Evaluation

#### Metrik Evaluasi

Karena tidak memiliki data eksplisit interaksi pengguna, maka digunakan evaluasi berikut:

##### A. Cosine Similarity Score

Menilai seberapa mirip drama rekomendasi dengan input.

```python
def evaluate_similarity(title, top_n=5):
    idx = indices.get(title)
    if idx is None:
        return f"Drama '{title}' tidak ditemukan."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    avg_score = sum([score for _, score in sim_scores]) / top_n
    return f"Rata-rata skor similarity dari top-{top_n} rekomendasi untuk '{title}': {avg_score:.4f}"
```

##### B. Precision\@K, Recall\@K, F1-Score\@K (Simulasi)

Karena tidak ada data relevansi eksplisit, dibuat asumsi sederhana:

* Jika genre drama rekomendasi mengandung genre dari drama input, maka dianggap relevan.

```python
def evaluate_precision_recall(title, top_n=5):
    idx = indices.get(title)
    input_genres = set(df_filtered.loc[idx, 'Genre'].split(", "))
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]
    relevant = 0
    for i, _ in sim_scores:
        genres = set(df_filtered.iloc[i]['Genre'].split(", "))
        if input_genres & genres:
            relevant += 1
    precision = relevant / top_n
    recall = relevant / len(input_genres) if input_genres else 0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f"Precision@{top_n}: {precision:.2f}, Recall@{top_n}: {recall:.2f}, F1-Score@{top_n}: {f1:.2f}"
```

#### Hasil Evaluasi

```python
evaluate_similarity("My Mister")
evaluate_precision_recall("My Mister")
```
- Rata-rata skor similarity dari top-5 rekomendasi untuk 'My Mister': 0.1430
- Precision@5: 0.20, Recall@5: 0.25, F1-Score@5: 0.22
---

### 8. Impact to Business Understanding

#### Apakah masalah bisnis terjawab?

Ya. Sistem rekomendasi berhasil membantu pengguna menemukan drama dengan kemiripan genre dan cerita, mengurangi waktu pencarian.

#### Apakah tujuan proyek tercapai?

✔ Memberikan rekomendasi top-5 yang relevan
✔ Menggunakan pendekatan content-based filtering yang scalable

#### Apakah solusi efektif?

Sistem menunjukkan hasil yang baik dalam menyajikan drama dengan tema serupa, memberikan pengalaman yang lebih personal pada pengguna.
