// Author: Vinicius Atsushi Sato Kawai <vinicius.kawai@unesp.br>
// Code implementation of Rank-based Hashing for Effective and Efficient Nearest Neighbor Search for Image Retrieval

/*
 *
 * Copyright (c) 2024, Vinicius Atsushi Sato Kawai (São Paulo State University)
 * All rights reserved.
 *
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include <fstream>

#define N_IMAGES 1360        // Dataset size
#define FEATURES 4096        // Dimensionality of the feature vectors
#define HASH_SIZE 1021       // Number of hash buckets used per feature dimension
#define P_DECAY 0.99         // Exponential decay factor used in the weighting function
#define N_NEIGHBORS 30       // Number of neighbors considered per feature during aggregation
#define TOP_N 1000           // Number of top candidates selected based on aggregated weights
#define TOP_K 80             // Number of top-ranked candidates selected for re-ranking with Euclidean distance (TOP_K <= TOP_N)


using namespace std;

struct Node {
    int id;
    float val;
    Node* next;
};
typedef Node* HashTable[HASH_SIZE];

int hash_index(float value) {
    int h = (int)(value * HASH_SIZE + 0.5);
    return min(h, HASH_SIZE - 1);
}

void insert_sorted_hash(HashTable& table, int id, float val) {
    int h = hash_index(val);
    Node* new_node = new Node{id, val, nullptr};
    if (!table[h] || val < table[h]->val) {
        new_node->next = table[h];
        table[h] = new_node;
        return;
    }
    Node* curr = table[h];
    while (curr->next && val >= curr->next->val)
        curr = curr->next;
    new_node->next = curr->next;
    curr->next = new_node;
}

void clear_hash(HashTable& table) {
    for (int i = 0; i < HASH_SIZE; ++i) {
        Node* current = table[i];
        while (current) {
            Node* tmp = current;
            current = current->next;
            delete tmp;
        }
        table[i] = nullptr;
    }
}

float* load_feat_matrix(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { cerr << "Unable to open features file." << endl; exit(1); }
    float* matrix = new float[N_IMAGES * FEATURES];
    for (int i = 0; i < N_IMAGES * FEATURES; ++i)
        fscanf(file, "%f,", &matrix[i]);
    fclose(file);
    return matrix;
}

void compute_and_normalize_ranked(float* matrix, vector<int>& rk_id, vector<float>& rk_val, vector<float>& min_val, vector<float>& max_val) {
    rk_id.resize(N_IMAGES * FEATURES);
    rk_val.resize(N_IMAGES * FEATURES);
    min_val.resize(FEATURES);
    max_val.resize(FEATURES);

    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        vector<pair<float, int>> temp(N_IMAGES);
        for (int i = 0; i < N_IMAGES; ++i)
            temp[i] = make_pair(matrix[i * FEATURES + f], i);
        sort(temp.begin(), temp.end());

        float minv = temp.front().first;
        float maxv = temp.back().first;
        min_val[f] = minv;
        max_val[f] = maxv;
        float range = maxv - minv;

        for (int i = 0; i < N_IMAGES; ++i) {
            rk_val[f * N_IMAGES + i] = (range > 0) ? (temp[i].first - minv) / range : 0.0;
            rk_id[f * N_IMAGES + i] = temp[i].second;
        }
    }
}

void build_hash_tables(vector<float>& rk_val, vector<int>& rk_id, vector<HashTable>& hash_tables) {
    for (int f = 0; f < FEATURES; ++f) {
        for (int i = 0; i < HASH_SIZE; ++i) hash_tables[f][i] = nullptr;
        for (int i = 0; i < N_IMAGES; ++i) {
            int img_id = rk_id[f * N_IMAGES + i];
            float val = rk_val[f * N_IMAGES + i];
            insert_sorted_hash(hash_tables[f], img_id, val);
        }
    }
}

void normalize_query(float* query, float* norm_query, const vector<float>& min_val, const vector<float>& max_val) {
    for (int f = 0; f < FEATURES; ++f) {
        float minv = min_val[f];
        float maxv = max_val[f];
        float range = maxv - minv;
        norm_query[f] = (range > 0) ? (query[f] - minv) / range : 0.0;
    }
}

void aggregate_from_hash(float* query, const vector<HashTable>& hash_tables, float* weights) {
    memset(weights, 0, sizeof(float) * N_IMAGES);

    for (int f = 0; f < FEATURES; ++f) {
        int h = hash_index(query[f]);
        int neighbors = 0;
        for (int delta = 0; delta < HASH_SIZE && neighbors < N_NEIGHBORS; ++delta) {
            for (int dir = -1; dir <= 1; dir += 2) {
                int bucket = h + delta * dir;
                if (bucket < 0 || bucket >= HASH_SIZE) continue;
                Node* node = hash_tables[f][bucket];
                int rank = 0;
                while (node && neighbors < N_NEIGHBORS) {
                    weights[node->id] += (N_NEIGHBORS - rank) * pow(P_DECAY, rank + 1);
                    node = node->next;
                    rank++;
                    neighbors++;
                }
            }
        }
    }
}

void rerank_by_euclidean(float* matrix, int* ids, int topk, int query_id) {
    vector<pair<float, int>> dist_id;
    for (int i = 0; i < topk; ++i) {
        int id = ids[i];
        float dist = 0.0f;
        for (int f = 0; f < FEATURES; ++f) {
            float diff = matrix[query_id * FEATURES + f] - matrix[id * FEATURES + f];
            dist += diff * diff;
        }
        dist_id.emplace_back(sqrt(dist), id);
    }
    sort(dist_id.begin(), dist_id.end());
    for (int i = 0; i < topk; ++i)
        ids[i] = dist_id[i].second;
}

int main() {
    float* feat_matrix = load_feat_matrix("./files/feat-matrix.txt");
    vector<int> rk_id;
    vector<float> rk_val, min_val, max_val;
    compute_and_normalize_ranked(feat_matrix, rk_id, rk_val, min_val, max_val);

    vector<HashTable> hash_tables(FEATURES);
    build_hash_tables(rk_val, rk_id, hash_tables);

    float* weights = new float[N_IMAGES];
    int* topN_ids = new int[TOP_N];
    float* norm_query = new float[FEATURES];

    ofstream out_ids("rks.txt");
    ofstream out_dists("dists.txt");

    for (int q = 0; q < N_IMAGES; ++q) {
        normalize_query(&feat_matrix[q * FEATURES], norm_query, min_val, max_val);
        aggregate_from_hash(norm_query, hash_tables, weights);

        vector<pair<float, int>> w_id;
        for (int i = 0; i < N_IMAGES; ++i)
            w_id.emplace_back(weights[i], i);
        partial_sort(w_id.begin(), w_id.begin() + TOP_N, w_id.end(), greater<>());
        for (int i = 0; i < TOP_K; ++i)
            topN_ids[i] = w_id[i].second;

        rerank_by_euclidean(feat_matrix, topN_ids, TOP_K, q);

        for (int i = 0; i < TOP_K; ++i) {
            int id = topN_ids[i];
            float dist = 0.0;
            for (int f = 0; f < FEATURES; ++f) {
                float diff = feat_matrix[q * FEATURES + f] - feat_matrix[id * FEATURES + f];
                dist += diff * diff;
            }
            out_ids << id << " ";
            out_dists << sqrt(dist) << " ";
        }
        out_ids << "\n";
        out_dists << "\n";
        if (q % 100 == 0) printf("Query progress: %d / %d \n", q, N_IMAGES);
    }

    out_ids.close();
    out_dists.close();

    delete[] weights;
    delete[] topN_ids;
    delete[] norm_query;
    delete[] feat_matrix;
    for (int f = 0; f < FEATURES; ++f) clear_hash(hash_tables[f]);
    return 0;
}
