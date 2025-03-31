# Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval

Implementation of the method described in the paper:
> [**Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval**](https://dl.acm.org/doi/abs/10.1145/3659580)  
> **ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), 2024**

---

## 📌 Authors

- [Vinicius Atsushi Sato Kawai](https://github.com/ViniciusAtsushi)
- [Lucas Pascotti Valem](http://www.lucasvalem.com)
- [Alexandro Baldassin](https://github.com/baldas)
- [Edson Borin](https://www.ic.unicamp.br/~edson/)
- [Daniel Carlos Guimarães Pedronette](http://www.ic.unicamp.br/~dcarlos/)
- [Longin Jan Latecki](https://cis.temple.edu/~latecki/)

Department of Statistics, Applied Mathematics and Computing  
**Universidade Estadual Paulista (UNESP)**, Rio Claro, Brazil

---

## 🧠 Overview

This repository provides a C++ implementation of a rank-based hashing method designed for effective and efficient nearest neighbor search in image retrieval tasks. The approach is **unsupervised, data-independent**, and supports both:

- Internal retrieval (searching within the indexed dataset), and
- External querying (evaluating new feature vectors at runtime).

There are two core implementations provided:

### 🔹 Version 1 (Direct Access)
- Uses feature-wise sorting to build ranked indices.
- During querying, access is performed directly through rank positions.
- Simpler structure without explicit hash tables.

### 🔹 Version 2 (Hash-based Access)
- Builds sorted hash tables for each feature dimension.
- Each query accesses nearest neighbors using bucket-based collision.
- Supports approximate but efficient retrieval.

Both implementations have variants that support only indexing (v1, v2) and indexing plus querying external data (v1.1, v2.1).

---

## 📥 Dataset Format

### 1. `feat-matrix.txt`
- Mandatory input file for **all versions**.
- Contains one feature vector per image.
- Format: one line per image, with comma-separated float values.

Example:
```
0.0123,0.2345,...
0.1111,0.2222,...
...
```

> The number of values per line (i.e., feature dimensionality) must match the constant `FEATURES` defined in the source code.

### 2. `query_features.txt`
- Required for the **indexing versions** (v1.1 and v2.1).
- Same format as `feat-matrix.txt`.
- Used to test external queries against the pre-built index.

---

## ⚙️ Compilation

```bash
cd src/
g++ rank_based_hash_v1.cpp -O3 -fopenmp -o hash_v1
g++ rank_based_hash_v2.cpp -O3 -fopenmp -o hash_v2
g++ rank_based_hash_indexing_v1.1.cpp -O3 -fopenmp -o hash_index_v1
g++ rank_based_hash_indexing_v2.1.cpp -O3 -fopenmp -o hash_index_v2
```

> Requires a C++ compiler with OpenMP support (e.g., `g++`).

---

## 🚀 Execution

### Version 1: Access with ranking only
```bash
./hash_v1
# Output: rks.txt, dists.txt
```

### Version 2: Access using hash tables
```bash
./hash_v2
# Output: rks.txt, dists.txt
```

### Version 1.1: External queries (ranking-based)
```bash
./hash_index_v1 ../files/query_features.txt
# Output: query_rks.txt, query_dists.txt
```

### Version 2.1: External queries (hash-based)
```bash
./hash_index_v2 ../files/query_features.txt
# Output: query_rks.txt, query_dists.txt
```

---

## ⚙️ Parameters

Defined in the header of each `.cpp` file:
```cpp
#define N_IMAGES 1360       // Number of database images
#define FEATURES 4096       // Feature dimensionality
#define HASH_SIZE 1021      // Number of buckets per hash table (V2 only)
#define P_DECAY 0.99        // Weight decay for rank voting
#define N_NEIGHBORS 30      // Nearest neighbors per feature
#define TOP_N 1000          // Number of candidates selected by aggregation
#define TOP_K 80            // Final candidates re-ranked by Euclidean distance (TOP_K <= TOP_N)
```

---

## 📂 Folder Structure
```
RankBasedHashing/
├── src/
│   ├── rank_based_hash_v1.cpp
│   ├── rank_based_hash_v2.cpp
│   ├── rank_based_hash_indexing_v1.1.cpp
│   └── rank_based_hash_indexing_v2.1.cpp
├── files/
│   └── feat-matrix.txt
│   └── query_features.txt
├── README.md
```

> Output files like `rks.txt` and `query_rks.txt` will be generated after execution and are not versioned in the repository.

---

## 📖 Citation

If you use this code in your research, please cite the following paper:

 ```latex
@article{10.1145/3659580,
author = {Kawai, Vinicius Sato and Valem, Lucas Pascotti and Baldassin, Alexandro and Borin, Edson and Pedronette, Daniel Carlos Guimar\~{a}es and Latecki, Longin Jan},
title = {Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval},
year = {2024},
issue_date = {October 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {20},
number = {10},
issn = {1551-6857},
url = {https://doi.org/10.1145/3659580},
doi = {10.1145/3659580},
abstract = {The large and growing amount of digital data creates a pressing need for approaches capable of indexing and retrieving multimedia content. A traditional and fundamental challenge consists of effectively and efficiently performing nearest-neighbor searches. After decades of research, several different methods are available, including trees, hashing, and graph-based approaches. Most of the current methods exploit learning to hash approaches based on deep learning. In spite of effective results and compact codes obtained, such methods often require a significant amount of labeled data for training. Unsupervised approaches also rely on expensive training procedures usually based on a huge amount of data. In this work, we propose an unsupervised data-independent approach for nearest neighbor searches, which can be used with different features, including deep features trained by transfer learning. The method uses a rank-based formulation and exploits a hashing approach for efficient ranked list computation at query time. A comprehensive experimental evaluation was conducted on seven public datasets, considering deep features based on CNNs and Transformers. Both effectiveness and efficiency aspects were evaluated. The proposed approach achieves remarkable results in comparison to traditional and state-of-the-art methods. Hence, it is an attractive and innovative solution, especially when costly training procedures need to be avoided.},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
month = sep,
articleno = {329},
numpages = {19},
keywords = {Rank, hashing, retrieval, image search, effectiveness, efficiency, unupervised}
}
```

---

## 📬 Contact

For questions or contributions, feel free to contact:
**Vinicius Atsushi Sato Kawai** — [vinicius.kawai@unesp.br](mailto:vinicius.kawai@unesp.br)
