# Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval
Implementation of 'Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval' (https://dl.acm.org/doi/abs/10.1145/3659580)

**Authors:** [Vinicius Atsushi Sato Kawai](https://github.com/ViniciusAtsushi), [Lucas Pascotti Valem](http://www.lucasvalem.com), [Alexandro Baldassin](https://github.com/baldas), [Edson Borin](https://www.ic.unicamp.br/~edson/), [Daniel Carlos Guimarães Pedronette](http://www.ic.unicamp.br/~dcarlos/), and [Longin Jan Latecki](https://cis.temple.edu/~latecki/)

Dept. of Statistic, Applied Math. and Computing, Universidade Estadual Paulista ([UNESP](http://www.rc.unesp.br/)), Rio Claro, Brazil

## Cite
If you use this implementation, please cite

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
