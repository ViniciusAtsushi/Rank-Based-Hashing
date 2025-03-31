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

#define N_IMAGES 25          // Dataset size
#define FEATURES 4096        // Dimensionality of the feature vectors
#define P_DECAY 0.99         // Exponential decay factor used in the weighting function
#define N_NEIGHBORS 15       // Number of neighbors considered per feature during aggregation
#define TOP_N 25             // Number of top candidates selected based on aggregated weights
#define TOP_K 10             // Number of top-ranked candidates selected for re-ranking with Euclidean distance (TOP_K <= TOP_N)


using namespace std;

// ------------------------ Step 1: Load the feature matrix ------------------------
float* load_feat_matrix(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) { cerr << "Unable to open features file." << endl; exit(1); }
    float* matrix = new float[N_IMAGES * FEATURES];
    for (int i = 0; i < N_IMAGES * FEATURES; ++i)
        fscanf(file, "%f,", &matrix[i]);
    fclose(file);
    return matrix;
}

// ------------------------ Step 2: Sort features by dimension ---------------------
void compute_ranked_features(float* matrix, vector<int>& rk_id, vector<float>& rk_val) {
    rk_id.resize(N_IMAGES * FEATURES);
    rk_val.resize(N_IMAGES * FEATURES);
    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        vector<pair<float, int>> temp(N_IMAGES);
        for (int i = 0; i < N_IMAGES; ++i)
            temp[i] = make_pair(matrix[i * FEATURES + f], i);
        sort(temp.begin(), temp.end());
        for (int i = 0; i < N_IMAGES; ++i) {
            rk_val[f * N_IMAGES + i] = temp[i].first;
            rk_id[f * N_IMAGES + i] = temp[i].second;
        }
    }
}

// ------------------------ Step 3: Apply min-max normalization --------------------
void normalize_features(vector<float>& rk_val) {
    #pragma omp parallel for
    for (int f = 0; f < FEATURES; ++f) {
        float minv = rk_val[f * N_IMAGES];
        float maxv = rk_val[f * N_IMAGES];
        for (int i = 1; i < N_IMAGES; ++i) {
            float v = rk_val[f * N_IMAGES + i];
            if (v < minv) minv = v;
            if (v > maxv) maxv = v;
        }
        float range = maxv - minv;
        if (range > 0) {
            for (int i = 0; i < N_IMAGES; ++i)
                rk_val[f * N_IMAGES + i] = (rk_val[f * N_IMAGES + i] - minv) / range;
        }
    }
}

// ------------------------ Step 4: Reorganize feature values by image -------------
void reorganize_features_by_image(vector<float>& rk_val, vector<int>& rk_id, float* output_matrix) {
    for (int f = 0; f < FEATURES; ++f)
        for (int i = 0; i < N_IMAGES; ++i) {
            int id = rk_id[f * N_IMAGES + i];
            output_matrix[id * FEATURES + f] = rk_val[f * N_IMAGES + i];
        }
}

// ------------------------ Step 5: Accumulate weights using ranking proximity -----
void aggregate_by_ranking(const vector<vector<int>>& rk_id, int query_id, float* weights) {
    memset(weights, 0, sizeof(float) * N_IMAGES);

    for (int f = 0; f < FEATURES; ++f) {
        for (int i = 0; i < N_IMAGES; ++i) {
            if (rk_id[f][i] == query_id) {
                int start = max(0, i - N_NEIGHBORS / 2);
                int end = min(N_IMAGES, start + N_NEIGHBORS);
                for (int j = start; j < end; ++j) {
                    int id = rk_id[f][j];
                    int pos = j - start;
                    float peso = (N_NEIGHBORS - pos) * pow(P_DECAY, pos + 1);
                    weights[id] += peso;
                }
                break;
            }
        }
    }
}

// ------------------------ Step 6: Re-rank top candidates using Euclidean distance -----
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

// ----------------------------------------------------------------------------------------
int main() {
    float* feat_matrix = load_feat_matrix("../files/feat-matrix.txt");

    vector<int> rk_id_flat;
    vector<float> rk_val;
    compute_ranked_features(feat_matrix, rk_id_flat, rk_val);
    normalize_features(rk_val);

    float* all_images = new float[N_IMAGES * FEATURES]();
    reorganize_features_by_image(rk_val, rk_id_flat, all_images);

    vector<vector<int>> rk_id(FEATURES, vector<int>(N_IMAGES));
    for (int f = 0; f < FEATURES; ++f)
        for (int i = 0; i < N_IMAGES; ++i)
            rk_id[f][i] = rk_id_flat[f * N_IMAGES + i];

    float* weights = new float[N_IMAGES];
    int* topN_ids = new int[TOP_N];

    ofstream out_ids("rks.txt");
    ofstream out_dists("dists.txt");

    for (int q = 0; q < N_IMAGES; ++q) {
        aggregate_by_ranking(rk_id, q, weights);

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
    delete[] feat_matrix;
    delete[] all_images;
    return 0;
}