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

int hash_index(float value) {
    int h = (int)(value * HASH_SIZE + 0.5);
    return min(h, HASH_SIZE - 1);
}

void insert_sorted_hash(vector<vector<Node*>>& table, int f, int id, float val) {
    int h = hash_index(val);
    Node* new_node = new Node{id, val, nullptr};
    if (!table[f][h] || val < table[f][h]->val) {
        new_node->next = table[f][h];
        table[f][h] = new_node;
        return;
    }
    Node* curr = table[f][h];
    while (curr->next && val >= curr->next->val)
        curr = curr->next;
    new_node->next = curr->next;
    curr->next = new_node;
}

void clear_hash(vector<vector<Node*>>& table) {
    for (int f = 0; f < FEATURES; ++f) {
        for (int i = 0; i < HASH_SIZE; ++i) {
            Node* current = table[f][i];
            while (current) {
                Node* tmp = current;
                current = current->next;
                delete tmp;
            }
            table[f][i] = nullptr;
        }
    }
}

float* load_feat_matrix(const char* filename, int& n_images) {
    FILE* file = fopen(filename, "r");
    if (!file) { cerr << "Unable to open features file." << endl; exit(1); }
    vector<float> values;
    float temp;
    while (fscanf(file, "%f,", &temp) == 1) values.push_back(temp);
    fclose(file);
    n_images = values.size() / FEATURES;
    float* matrix = new float[values.size()];
    memcpy(matrix, values.data(), sizeof(float) * values.size());
    return matrix;
}

void compute_and_normalize_ranked(float* matrix, int n_images, vector<int>& rk_id, vector<float>& rk_val, vector<float>& min_val, vector<float>& max_val) {
    rk_id.resize(n_images * FEATURES);
    rk_val.resize(n_images * FEATURES);
    min_val.resize(FEATURES);
    max_val.resize(FEATURES);

    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        vector<pair<float, int>> temp(n_images);
        for (int i = 0; i < n_images; ++i)
            temp[i] = make_pair(matrix[i * FEATURES + f], i);
        sort(temp.begin(), temp.end());

        float minv = temp.front().first;
        float maxv = temp.back().first;
        min_val[f] = minv;
        max_val[f] = maxv;
        float range = maxv - minv;

        for (int i = 0; i < n_images; ++i) {
            rk_val[f * n_images + i] = (range > 0) ? (temp[i].first - minv) / range : 0.0;
            rk_id[f * n_images + i] = temp[i].second;
        }
    }
}

void build_hash_tables(vector<float>& rk_val, vector<int>& rk_id, int n_images, vector<vector<Node*>>& hash_tables) {
    hash_tables.assign(FEATURES, vector<Node*>(HASH_SIZE, nullptr));
    for (int f = 0; f < FEATURES; ++f)
        for (int i = 0; i < n_images; ++i) {
            int img_id = rk_id[f * n_images + i];
            float val = rk_val[f * n_images + i];
            insert_sorted_hash(hash_tables, f, img_id, val);
        }
}

void update_min_max_for_query(const float* query, vector<float>& min_val, vector<float>& max_val) {
    for (int f = 0; f < FEATURES; ++f) {
        if (query[f] < min_val[f]) min_val[f] = query[f];
        if (query[f] > max_val[f]) max_val[f] = query[f];
    }
}

void normalize_query(const float* query, float* norm_query, const vector<float>& min_val, const vector<float>& max_val) {
    for (int f = 0; f < FEATURES; ++f) {
        float range = max_val[f] - min_val[f];
        norm_query[f] = (range > 0) ? (query[f] - min_val[f]) / range : 0.0;
    }
}

void aggregate_from_hash(const float* query, const vector<vector<Node*>>& hash_tables, float* weights, int n_images) {
    memset(weights, 0, sizeof(float) * n_images);
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

void rerank_by_euclidean(float* matrix, int* ids, float* dists, int topk, const float* query) {
    vector<pair<float, int>> dist_id;
    for (int i = 0; i < topk; ++i) {
        int id = ids[i];
        float dist = 0.0f;
        for (int f = 0; f < FEATURES; ++f) {
            float diff = query[f] - matrix[id * FEATURES + f];
            dist += diff * diff;
        }
        dist_id.emplace_back(sqrt(dist), id);
    }
    sort(dist_id.begin(), dist_id.end());
    for (int i = 0; i < topk; ++i) {
        dists[i] = dist_id[i].first;
        ids[i] = dist_id[i].second;
    }
}

void query_image_hash(const float* raw_query, const vector<float>& min_val, const vector<float>& max_val,
                      const vector<vector<Node*>>& hash_tables, float* feat_matrix, int n_images,
                      ofstream& out_rk, ofstream& out_dist) {
    float* norm_query = new float[FEATURES];
    normalize_query(raw_query, norm_query, min_val, max_val);

    float* weights = new float[n_images];
    aggregate_from_hash(norm_query, hash_tables, weights, n_images);

    vector<pair<float, int>> w_id;
    for (int i = 0; i < n_images; ++i)
        w_id.emplace_back(weights[i], i);
    partial_sort(w_id.begin(), w_id.begin() + TOP_N, w_id.end(), greater<>());

    int topN_ids[TOP_N];
    float topN_dists[TOP_K];
    for (int i = 0; i < TOP_K; ++i)
        topN_ids[i] = w_id[i].second;

    rerank_by_euclidean(feat_matrix, topN_ids, topN_dists, TOP_K, raw_query);

    for (int i = 0; i < TOP_K; ++i) {
        out_rk << topN_ids[i] << " ";
        out_dist << topN_dists[i] << " ";
    }
    out_rk << "\n";
    out_dist << "\n";

    delete[] norm_query;
    delete[] weights;
}

int main(int argc, char** argv) {
    int n_images;
    float* feat_matrix = load_feat_matrix("./files/feat-matrix.txt", n_images);

    vector<int> rk_id;
    vector<float> rk_val, min_val, max_val;
    compute_and_normalize_ranked(feat_matrix, n_images, rk_id, rk_val, min_val, max_val);

    vector<vector<Node*>> hash_tables;
    build_hash_tables(rk_val, rk_id, n_images, hash_tables);

    printf("Hash index built. Ready for external queries.\n");

    if (argc > 1) {
        int n_query_images;
        float* query_matrix = load_feat_matrix(argv[1], n_query_images);

        ofstream out_rk("query_rks.txt");
        ofstream out_dist("query_dists.txt");

        for (int q = 0; q < n_query_images; ++q) {
            update_min_max_for_query(&query_matrix[q * FEATURES], min_val, max_val);
            query_image_hash(&query_matrix[q * FEATURES], min_val, max_val, hash_tables, feat_matrix, n_images, out_rk, out_dist);
        }

        out_rk.close();
        out_dist.close();
        delete[] query_matrix;
    }

    delete[] feat_matrix;
    clear_hash(hash_tables);
    return 0;
}
