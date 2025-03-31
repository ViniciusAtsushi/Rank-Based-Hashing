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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <algorithm>
#include <fstream>

#define N_IMAGES 1360        // Dataset size
#define FEATURES 4096        // Dimensionality of the feature vectors
#define P_DECAY 0.99         // Exponential decay factor used in the weighting function
#define N_NEIGHBORS 30       // Number of neighbors considered per feature during aggregation
#define TOP_N 1000           // Number of top candidates selected based on aggregated weights
#define TOP_K 80             // Number of top-ranked candidates selected for re-ranking with Euclidean distance (TOP_K <= TOP_N)

using namespace std;

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

void compute_ranked_features(float* matrix, int n_images, vector<int>& rk_id, vector<float>& rk_val) {
    rk_id.resize(n_images * FEATURES);
    rk_val.resize(n_images * FEATURES);
    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        vector<pair<float, int>> temp(n_images);
        for (int i = 0; i < n_images; ++i)
            temp[i] = make_pair(matrix[i * FEATURES + f], i);
        sort(temp.begin(), temp.end());
        for (int i = 0; i < n_images; ++i) {
            rk_val[f * n_images + i] = temp[i].first;
            rk_id[f * n_images + i] = temp[i].second;
        }
    }
}

void normalize_features(vector<float>& rk_val, int n_images, vector<float>& min_val, vector<float>& max_val) {
    min_val.resize(FEATURES);
    max_val.resize(FEATURES);
    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        float minv = rk_val[f * n_images];
        float maxv = rk_val[f * n_images];
        for (int i = 1; i < n_images; ++i) {
            float v = rk_val[f * n_images + i];
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        float range = maxv - minv;
        min_val[f] = minv;
        max_val[f] = maxv;
        if (range > 0) {
            for (int i = 0; i < n_images; ++i)
                rk_val[f * n_images + i] = (rk_val[f * n_images + i] - minv) / range;
        }
    }
}

void update_min_max_for_query(const float* query, vector<float>& min_val, vector<float>& max_val) {
    for (int f = 0; f < FEATURES; ++f) {
        if (query[f] < min_val[f]) min_val[f] = query[f];
        if (query[f] > max_val[f]) max_val[f] = query[f];
    }
}

void build_rank_vectors(vector<float>& rk_val, vector<int>& rk_id, int n_images,
                        vector<vector<float>>& rk_val_by_feature, vector<vector<int>>& rk_id_by_feature) {
    rk_val_by_feature.resize(FEATURES);
    rk_id_by_feature.resize(FEATURES);
    for (int f = 0; f < FEATURES; ++f) {
        rk_val_by_feature[f].resize(n_images);
        rk_id_by_feature[f].resize(n_images);
        for (int i = 0; i < n_images; ++i) {
            rk_val_by_feature[f][i] = rk_val[f * n_images + i];
            rk_id_by_feature[f][i] = rk_id[f * n_images + i];
        }
    }
}

void aggregate_by_ranking(const vector<vector<int>>& rk_id, const vector<vector<float>>& rk_val, const float* query, float* weights, int n_images) {
    memset(weights, 0, sizeof(float) * n_images);
    for (int f = 0; f < FEATURES; ++f) {
        auto it = lower_bound(rk_val[f].begin(), rk_val[f].end(), query[f]);
        int pos = distance(rk_val[f].begin(), it);
        int start = max(0, pos - N_NEIGHBORS / 2);
        int end = min(n_images, start + N_NEIGHBORS);
        for (int j = start; j < end; ++j) {
            int id = rk_id[f][j];
            int rel_pos = j - start;
            weights[id] += (N_NEIGHBORS - rel_pos) * pow(P_DECAY, rel_pos + 1);
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

void query_image(const float* raw_query, const vector<float>& min_val, const vector<float>& max_val,
                 const vector<vector<int>>& rk_id, const vector<vector<float>>& rk_val,
                 float* feat_matrix, int n_images, ofstream& out_rk, ofstream& out_dist) {
    float* norm_query = new float[FEATURES];
    for (int f = 0; f < FEATURES; ++f) {
        float range = max_val[f] - min_val[f];
        norm_query[f] = (range > 0) ? (raw_query[f] - min_val[f]) / range : 0.0;
    }

    float* weights = new float[n_images];
    aggregate_by_ranking(rk_id, rk_val, norm_query, weights, n_images);

    vector<pair<float, int>> w_id;
    for (int i = 0; i < n_images; ++i)
        w_id.emplace_back(weights[i], i);
    partial_sort(w_id.begin(), w_id.begin() + TOP_N, w_id.end(), greater<>());

    int topN_ids[TOP_N];
    float topN_dists[TOP_K];  // ainda usamos apenas os TOP_K para reordenar e salvar

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

    vector<int> rk_id_flat;
    vector<float> rk_val_flat, min_val, max_val;
    compute_ranked_features(feat_matrix, n_images, rk_id_flat, rk_val_flat);
    normalize_features(rk_val_flat, n_images, min_val, max_val);

    vector<vector<int>> rk_id;
    vector<vector<float>> rk_val;
    build_rank_vectors(rk_val_flat, rk_id_flat, n_images, rk_val, rk_id);

    printf("Indexing complete. Ready for external queries.\n");

    if (argc > 1) {
        int n_query_images;
        float* query_matrix = load_feat_matrix(argv[1], n_query_images);

        ofstream out_rk("query_rks.txt");
        ofstream out_dist("query_dists.txt");

        for (int q = 0; q < n_query_images; ++q) {
            update_min_max_for_query(&query_matrix[q * FEATURES], min_val, max_val);
            query_image(&query_matrix[q * FEATURES], min_val, max_val, rk_id, rk_val, feat_matrix, n_images, out_rk, out_dist);
        }

        out_rk.close();
        out_dist.close();
        delete[] query_matrix;
    }

    delete[] feat_matrix;
    return 0;
}
