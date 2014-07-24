#ifndef MPI_FAST_LDA_H
#define MPI_FAST_LDA_H

#include <iostream>
#include <vector>
#include <map>
#include <tr1/unordered_map>
#include <string>
#include <fstream>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <Judy.h>
#include "mpi.h"
#include "str_pre.h"
#include "sparse_matrix.h"

using namespace std;
using namespace tr1;

class lda {
public:
  lda(const int &t, const double &a, const double &b, const int &i);
  ~lda();
  void read_words_dict(const string &words_dict_file);
  void read_trn_data(const string &trn_data_file);
  void init_est();
  void compute_s();
  void compute_s(const int &old_topic, const int &new_topic); 
  void compute_r(const int &doc);
  void compute_r(const int &doc, const int &old_topic, const int &new_topic); 
  void compute_q(const int &doc, const int &word);
  void compute_q(const int &doc, const int &word, const int &topic);
  void compute_deno_cache();
  void compute_deno_cache(const int &old_topic, const int &new_topic);
  void update_cache(const int &doc, const int &old_topic, const int &new_topic);
  void compute_s_cache();
  void compute_s_cache(const int &old_topic, const int &new_topic);
  void compute_r_cache(const int &doc);
  void compute_r_cache(const int &doc, const int &old_topic, const int &new_topic);
  void compute_q_cache(const int &doc);
  void compute_q_cache(const int &doc, const int &old_topic, const int &new_topic);
  void estimate();
  int sampling(const int &m, const int &n);
  void compute_theta();
  void compute_phi();
  void save_model_twords();
  double perplexity();
  int topic;
  double alpha;
  double beta;
  int iter;
  int rank;
  int np;
  int words_dict_sz;
  unordered_map<string, int> word_id_dict;
  unordered_map<int, string> id_word_dict;
  vector< vector<int> > trn_data;
  sparse_matrix nwk, nwkp, ndk; 
  double s, r, q, **theta, **phi, ab, bv, *deno_cache, *s_cache, *r_cache, *q_cache;
  int **z, **last_z;
  int *nksum, *ndsum;
  row_element *row;
};

lda::lda(const int &t, const double &a, const double &b, const int &i) {
  topic = t;
  alpha = a;
  beta = b;
  ab = alpha*beta;
  iter = i;
  words_dict_sz = 0;
  theta = NULL;
  phi = NULL;
  z = NULL;
  nksum = NULL;
  ndsum = NULL;
  row = (row_element *)malloc(topic*sizeof(row_element));
  deno_cache = (double *)malloc(topic*sizeof(double));
  s_cache = (double *)malloc(topic*sizeof(double));
  r_cache = (double *)malloc(topic*sizeof(double));
  q_cache = (double *)malloc(topic*sizeof(double));
  for (int i = 0; i < topic; ++i) {
    s_cache[i] = 0;
    r_cache[i] = 0;
    q_cache[i] = 0;
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
}
lda::~lda() {
  int M = trn_data.size();
  int K = topic;
  int V = words_dict_sz;
  if (theta != NULL) {
    for (int i = 0; i < M; ++i) free(theta[i]);
    free(theta);
  }
  if (phi != NULL) {
    for (int i = 0; i < K; ++i) free(phi[i]);
    free(phi);
  }
  if (z != NULL) {
    for (int i = 0; i < M; ++i) free(z[i]);
    free(z);
  }
  if (nksum != NULL) free(nksum);
  if (ndsum != NULL) free(ndsum);
  if (row != NULL) free(row);
  if (deno_cache != NULL) free(deno_cache);
  if (s_cache != NULL) free(s_cache);
  if (r_cache != NULL) free(r_cache);
  if (q_cache != NULL) free(q_cache);
}
void lda::read_words_dict(const string &words_dict_file) {
  int word_id = 0;
  ifstream in;
  in.open(words_dict_file.c_str(), ios::in);
  string line;
  while (1) {
    getline(in, line);
    if (in.eof()) break;
    unordered_map<string, int>::iterator it = word_id_dict.find(line);
    if (it == word_id_dict.end()) {
      word_id_dict[line] = word_id;
      id_word_dict[word_id] = line;
      ++word_id;
    }
  }
  in.close();
  words_dict_sz = word_id;
  std::cout << "words_dict_sz:" << words_dict_sz << std::endl;
  bv = beta*words_dict_sz;
}
void lda::read_trn_data(const string &trn_data_file) {
  ifstream in;
  in.open(trn_data_file.c_str(), ios::in);
  string line;
  while (1) {
    getline(in, line);
    if (in.eof()) break;
    vector<string> fields;
    vector<int> data;
    StrictSplit(line, " ", &fields);
    for (int i = 0; i < fields.size(); ++i) {
      if (fields[i] == "") continue;
      data.push_back(word_id_dict[fields[i]]);
    }
    trn_data.push_back(data);
  }
  std::cout << "train data size: " << trn_data.size() << std::endl;
  /*for (int i = 0; i < trn_data.size(); ++i) {
    int N = trn_data[i].size();
    for (int j = 0; j < N; j++) {
      std::cout << id_word_dict[trn_data[i][j]] << " ";
    }
    std::cout << std::endl;
  }*/
  in.close();
}
void lda::init_est() {
  std::cout << "init est" << std::endl;
  std::cout << "words_dict_sz:" << words_dict_sz << std::endl;
  int M = trn_data.size();
  z = (int **)malloc(M*sizeof(int*));
  for (int m = 0; m < M; ++m) {
    int N = trn_data[m].size();
    z[m] = (int *)malloc(N*sizeof(N));
  }
  nksum = (int *)malloc(topic*sizeof(int));
  for (int k = 0; k < topic; k++) nksum[k] = 0;
  ndsum = (int *)malloc(M*sizeof(int));
  for (int m = 0; m < M; m++) ndsum[m] = 0;
  nwk.row = words_dict_sz;
  nwk.col = topic;
  nwk.init_matrix();
  ndk.row = M;
  ndk.col = topic;
  ndk.init_matrix();
  nwkp.row = words_dict_sz;
  nwkp.col = topic;
  nwkp.init_matrix();
  srandom((int)time(0));
  for (int m = 0; m < M; ++m) {
    int N = trn_data[m].size();
    for (int n = 0; n < N; ++n) {
      int w = trn_data[m][n];
      z[m][n] = int(rand()/(double)(RAND_MAX)*topic);
      int w_topic = z[m][n];
      nwk.add(w, w_topic, 1);
      ndk.add(m, w_topic, 1);
      nksum[w_topic] += 1;
    }
    ndsum[m] += N;
  }
  int K = topic;
  theta = (double **)malloc(M*sizeof(double*));
  for (int i = 0; i < M; ++i) {
    theta[i] = (double *)malloc(K*sizeof(double));
  }
  int V = words_dict_sz;
  phi = (double **)malloc(K*sizeof(double*));
  for (int i = 0; i < K; ++i) {
    phi[i] = (double*)malloc(V*sizeof(double));
  }
}
inline void lda::compute_s() {
  int K = topic;
  s = 0;
  compute_s_cache();
  for (int k = 0; k < K; ++k) {
    s += s_cache[k];
  }
}
inline void lda::compute_s(const int &old_topic, const int &new_topic) {
  s -= ab/(deno_cache[old_topic]+1);
  s -= ab/(deno_cache[new_topic]-1);
  s += ab/(deno_cache[old_topic]);
  s += ab/(deno_cache[new_topic]);
}
inline void lda::compute_s_cache() {
  int K = topic;
  for (int k = 0; k < K; ++k) {
    s_cache[k] = ab/(deno_cache[k]);
  }
}
inline void lda::compute_s_cache(const int &old_topic, const int &new_topic) {
  s_cache[old_topic] = ab/(deno_cache[old_topic]);
  s_cache[new_topic] = ab/(deno_cache[new_topic]);
}
inline void lda::compute_r(const int &doc) {
  compute_r_cache(doc);
  r = 0;
  int cnt;
  Word_t c, next_c, val;
  bool ret = ndk.get_row_first(val, c, doc);
  r += r_cache[c];
  while (1) {
    bool ret = ndk.get_row_next(val, next_c, doc, c);
    if (!ret) break;
    r += r_cache[next_c];
    c = next_c;
  }
}
inline void lda::compute_r_cache(const int &doc) {
  Word_t c, next_c, val;
  bool ret = ndk.get_row_first(val, c, doc);
  r_cache[c] = val*beta/deno_cache[c];
  while (1) {
    bool ret = ndk.get_row_next(val, next_c, doc, c);
    if (!ret) break;
    r_cache[next_c] = val*beta/deno_cache[next_c];
    c = next_c;
  }
}
inline void lda::compute_r(const int &doc, const int &old_topic, const int &new_topic) {
  r -= (ndk.get(doc, old_topic)+1)*beta/(deno_cache[old_topic]+1);
  r -= (ndk.get(doc, new_topic)-1)*beta/(deno_cache[new_topic]-1);
  r += ndk.get(doc, old_topic)*beta/(deno_cache[old_topic]);
  r += ndk.get(doc, new_topic)*beta/(deno_cache[new_topic]);
}
inline void lda::compute_r_cache(const int &doc, const int &old_topic, const int &new_topic) {
  r_cache[old_topic] = ndk.get(doc, old_topic)*beta/(deno_cache[old_topic]);
  r_cache[new_topic] = ndk.get(doc, new_topic)*beta/(deno_cache[new_topic]);
}
inline void lda::compute_q(const int &doc, const int &word) {
  compute_q_cache(doc);
  q = 0;
  int cnt; 
  double q0, q1, q2, q3;
  bool ret = nwk.get_row(cnt, row, word);
  int k = 0;
  for (; (k+4) < cnt; k+=4) {
    q0 += q_cache[row[k].col]*row[k].val;
    q1 += q_cache[row[k+1].col]*row[k+1].val;
    q2 += q_cache[row[k+2].col]*row[k+2].val;
    q3 += q_cache[row[k+3].col]*row[k+3].val;
  }
  q = (q0+q1+q2+q3);
  for (; k < cnt; ++k) {
    q += q_cache[row[k].col]*row[k].val;
  }
}
inline void lda::compute_q(const int &doc, const int &word, const int &curr_topic) {
  q_cache[curr_topic] = (alpha+ndk.get(doc, curr_topic))/(deno_cache[curr_topic]);
  q = 0;
  Word_t val, c, next_c;
  bool ret = nwk.get_row_first(val, c, word);
  if (ret == false) {
    return;
  }
  q += q_cache[c]*val;
  while (1) {
    bool ret = nwk.get_row_next(val, next_c, word, c);
    if (!ret) break;
    q += q_cache[next_c]*val;
    c = next_c;
  }
}
inline void lda::compute_q_cache(const int &doc) {
  for (int k = 0; k < topic; ++k) {
    q_cache[k] = alpha/(deno_cache[k]);
  }
  Word_t val, c, next_c;
  bool ret = ndk.get_row_first(val, c, doc);
  q_cache[c] += val/deno_cache[c];
  while (1) {
    bool ret = ndk.get_row_next(val, next_c, doc, c);
    if (!ret) break;
    q_cache[next_c] += val/deno_cache[next_c];
    c = next_c;
  }

}
inline void lda::compute_q_cache(const int &doc, const int &old_topic, const int &new_topic) {
  q_cache[old_topic] = (alpha+ndk.get(doc, old_topic))/(deno_cache[old_topic]);
  q_cache[new_topic] = (alpha+ndk.get(doc, new_topic))/(deno_cache[new_topic]);
}
inline void lda::compute_deno_cache() {
  for (int k = 0; k < topic; ++k) {
    deno_cache[k] = bv+nksum[k];
  }
}
inline void lda::compute_deno_cache(const int &old_topic, const int &new_topic) {
  deno_cache[old_topic] = bv+nksum[old_topic];
  deno_cache[new_topic] = bv+nksum[new_topic];
}
inline void lda::update_cache(const int &doc, const int &old_topic, const int &new_topic) {
  compute_deno_cache(old_topic, new_topic);
  compute_s(old_topic, new_topic);
  compute_s_cache(old_topic, new_topic);
  compute_r(doc, old_topic, new_topic);
  compute_r_cache(doc, old_topic, new_topic);
  compute_q_cache(doc, old_topic, new_topic);
}
void lda::estimate() {
  compute_deno_cache();
  compute_s();
  int old_topic, new_topic;
  for (int i = 0; i < iter; ++i) {
    std::cout << "iteration: " << i << std::endl;
    int M = trn_data.size();
    for (int m = 0; m < M; ++m) {
      //std::cout << "iter" << i << " "<< "doc:" << m << std::endl;
      compute_r(m);
      compute_q_cache(m);
      int N = trn_data[m].size();
      for (int n = 0; n < N; ++n) {
        old_topic = z[m][n];
        new_topic = sampling(m, n);
        z[m][n] = new_topic;
        if (old_topic != new_topic) {
          update_cache(m, old_topic, new_topic);
        }
      }
    }
    //std::cout << "iteration: " << i << " " << perplexity() << std::endl;
  }
}
inline int lda::sampling(const int &m, const int &n) {
  int curr_topic = z[m][n];
  int w = trn_data[m][n];
  int K = topic;
  int V = words_dict_sz;
  nwk.sub(w, curr_topic, 1);
  ndk.sub(m, curr_topic, 1);
  nksum[curr_topic] -= 1;
  ndsum[m] -= 1;
  compute_q(m, w, curr_topic);
  double u = rand()/(double)(RAND_MAX)*(s+r+q);
  double curr_sum;
  int cnt = 0;
  //Word_t val = 0, c, next_c;
  if (u >= s + r) {
    curr_sum = s + r;
    Word_t c, val = 0, next_c;
    bool ret = nwk.get_row_first(val, c, w);
    curr_sum += q_cache[c]*val;
    while (true) {
      if (curr_sum > u) {
        curr_topic = c;
        break;
      }
      bool ret = nwk.get_row_next(val, next_c, w, c);
      if (!ret) break;
      curr_sum += q_cache[next_c]*val;
      c = next_c;
    }
  } else if (u >= s && u < s+r) {
    curr_sum = s;
    Word_t c, val = 0, next_c;
    bool ret = ndk.get_row_first(val, c, m);
    curr_sum += r_cache[c]*val;
    while (true) {
      if (curr_sum > u) {
        curr_topic = c;
        break;
      }
      bool ret = ndk.get_row_next(val, next_c, m, c);
      if (!ret) break;
      curr_sum += r_cache[next_c]*val;
      c = next_c;
    }
  } else if (u < s) {
    curr_sum = 0;
    for (int k = 0; k < K; ++k) {
      curr_sum += s_cache[k];
      if (curr_sum > u) {
        curr_topic = k;
        break;
      }
    }
  }
  nwk.add(w, curr_topic, 1);
  ndk.add(m, curr_topic, 1);
  nksum[curr_topic] += 1;
  ndsum[m] += 1;
  return curr_topic;
}
void lda::compute_theta() {
  int M = trn_data.size();
  int K = topic;
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) {
      theta[m][k] = (ndk.get(m,k)+alpha)/(ndsum[m]+K*alpha);
    }
  }
}
void lda::compute_phi() {
  int K = topic;
  int V = words_dict_sz;
  for (int k = 0; k < K; ++k) {
    for (int w = 0; w < V; ++w) {
      phi[k][w] =  (nwk.get(w,k)+beta)/(nksum[k]+V*beta);
    }
  }
}
int compare(const void *a, const void *b) {
  double sub =  (*(std::pair<int, double>*)b).second - (*(std::pair<int, double>*)a).second;
  if (sub > 0) {
    return 1;
  } else if (sub < 0) {
    return -1;
  } else {
    return 0;
  }
}
void lda::save_model_twords() {
  int K = topic;
  int V = words_dict_sz;
  for (int k = 0; k < K; ++k) {
    std::pair<int, double> *word_prob_lst = (std::pair<int, double>*)malloc(V*sizeof(std::pair<int, double>));
    for (int v = 0; v < V; ++v) {
      word_prob_lst[v] = std::make_pair(v, phi[k][v]);
    }
    qsort(word_prob_lst, V, sizeof(std::pair<int, double>), compare);
    std::cout << "Topic: " << k << endl;
    for (int i = 0; i < 10; ++i) {
      std::cout << "\t" << id_word_dict[word_prob_lst[i].first] << " " << word_prob_lst[i].second<< endl;
    }
  }
}
double lda::perplexity() {
  compute_theta();
  compute_phi();
  double p1 = 0, p2, d = 0;
  int M = trn_data.size();
  for (int m = 0; m < M; ++m) {
    int N = trn_data[m].size();
    d += N;
    for (int n = 0; n < N; ++n) {
      p2 = 0;
      for (int k = 0; k < topic; ++k) {
        p2 += phi[k][trn_data[m][n]]*theta[m][k];
        p2 += phi[k][trn_data[m][n]];
      }
      p2 = log(p2);
      p1 += p2;
    }
  }
  return exp(-1*p1/d);
}
void test() {
  std::pair<int, double> *lst = (std::pair<int, double>*)malloc(10*sizeof(std::pair<int, double>));
  for (int i = 0; i < 10; ++i) {
    lst[i] = std::make_pair(i, i);
  }
  qsort((void*)lst, 10, sizeof(std::pair<int, double>), compare);
  for (int i = 0; i < 10; ++i) {
    std::cout << lst[i].first << " " << lst[i].second << std::endl;
  }
}
#endif
