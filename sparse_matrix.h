#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include <vector>
#include <algorithm>
#include <utility>
#include "Judy.h"

typedef struct {
  Word_t col;
  Word_t val;
} row_element;

typedef struct {
  Word_t row;
  Word_t col;
  Word_t val;
} mat_element; 

class sparse_matrix {
public:
  Pvoid_t *Pmatrix;
  int row, col;
  sparse_matrix(const int &r, const int &c);
  sparse_matrix();
  ~sparse_matrix();
  void init_matrix();
  Word_t get(const Word_t &r, const Word_t &c);
  void set(const Word_t &r, const Word_t &c, const Word_t &v);
  void add(const Word_t &r, const Word_t &c, const Word_t &v);
  void sub(const Word_t &r, const Word_t &c, const Word_t &v);
  std::vector< std::pair<Word_t, Word_t> > get_row(const Word_t &r);
  bool get_row(int &cnt, row_element *curr_row, const Word_t &r);
  bool get_row_first(Word_t &val, Word_t &c, const Word_t &r);
  bool get_row_next(Word_t &val, Word_t &next_c, const Word_t &r, const Word_t &c);
  int cnt();
  int row_cnt(const Word_t &r);
};

sparse_matrix::sparse_matrix(const int &r, const int &c) {
    //Pmatrix = (Pvoid_t) NULL;
    row = r;
    col = c;
    Pmatrix = (Pvoid_t *)malloc(row*sizeof(Pvoid_t));
}
sparse_matrix::sparse_matrix() {
  Pmatrix = (Pvoid_t*) NULL;
}
void sparse_matrix::init_matrix() {
  if (Pmatrix == NULL) {
    std::cout << "init matrix" << std::endl;
    std::cout << "matrix row:" << row << std::endl;
    Pmatrix = (Pvoid_t *)malloc(row*sizeof(Pvoid_t));
  }
}
sparse_matrix::~sparse_matrix() {
  int Rc_word;
  //if (Pmatrix != NULL) JLFA(Rc_word, Pmatrix);
  if (Pmatrix != NULL) {
    for (int i = 0; i < row; i++) {
      JLFA(Rc_word, Pmatrix[i]);
    }
    free(Pmatrix);
  }
}
inline Word_t sparse_matrix::get(const Word_t &r, const Word_t &c) {
    Pvoid_t PValue;
    //Word_t Index = r*col + c;
    JLG(PValue, Pmatrix[r], c);
    if (PValue == PJERR || PValue == NULL) return 0;
    else {
        return *(static_cast<Word_t*>(PValue));
    }
}
inline void sparse_matrix::set(const Word_t &r, const Word_t &c, const Word_t &v) {
  Pvoid_t PValue;
  //Word_t Index = r*col + c;
  JLI(PValue, Pmatrix[r], c);
  *(static_cast<Word_t*>(PValue)) = v;
}
inline void sparse_matrix::add(const Word_t &r, const Word_t &c, const Word_t &v) {
  Pvoid_t PValue;
  //Word_t Index = r*col + c;
  JLG(PValue, Pmatrix[r], c);
  if (PValue == NULL) {
    JLI(PValue, Pmatrix[r], c);
    *(static_cast<Word_t*>(PValue)) = v;
  }
  else {
    int Rc_int;
    if ((*(static_cast<Word_t*>(PValue))+v) == 0 ) {
      JLD(Rc_int, Pmatrix[r], c);
    }else {
      *(static_cast<Word_t*>(PValue)) += v;
    }
  }
}
inline void sparse_matrix::sub(const Word_t &r, const Word_t &c, const Word_t &v) {
  Pvoid_t PValue;
  //Word_t Index = r*col + c;
  JLG(PValue, Pmatrix[r], c);
  if (PValue == NULL) {
    JLI(PValue, Pmatrix[r], c);
    *(static_cast<Word_t*>(PValue)) = -v;
  }
  else {
    int Rc_int;
    //*(static_cast<Word_t*>(PValue)) -= v;
    //if ((*(static_cast<Word_t*>(PValue)) == -v)) {
    if (((*(static_cast<Word_t*>(PValue)))-v) == 0) {
      //printf("value : %d v : %d\n", *(static_cast<Word_t*>(PValue)), v);
      JLD(Rc_int, Pmatrix[r], c);
    } else {
      *(static_cast<Word_t*>(PValue)) -= v;
    }
  }
}
inline std::vector< std::pair<Word_t, Word_t> > sparse_matrix::get_row(const Word_t &r) {
  //Word_t start_idx = r*col;
  //Word_t end_idx = (r+1)*col - 1;
  int Rc_word;
  JLC(Rc_word, Pmatrix[r], 0, -1);
  std::vector< std::pair<Word_t, Word_t> > res;
  Pvoid_t PValue;
  Word_t Index = 0;
  JLF(PValue, Pmatrix, Index);
  if (PValue == NULL) return res;
  res.push_back(std::make_pair(Index, *(static_cast<Word_t*>(PValue))));
  for (int i = 0; i < Rc_word-1; ++i) {
    JLN(PValue, Pmatrix[r], Index);
    if (PValue != NULL) {
      res.push_back(std::make_pair(Index, *(static_cast<Word_t*>(PValue))));
    }
  }
  return res;
}
inline int sparse_matrix::cnt() {
  int total_cnt = 0, cnt;
  //JLC(cnt, Pmatrix, 0, -1);
  for (int i = 0; i < row; ++i) {
    JLC(cnt, Pmatrix[i], 0, -1);
    total_cnt += cnt;
  }
  return total_cnt;
}
inline int sparse_matrix::row_cnt(const Word_t &r) {
  //Word_t start_idx = r*col;
  //Word_t end_idx = (r+1)*col - 1;
  int Rc_word;
  JLC(Rc_word, Pmatrix[r], 0, -1);
  return Rc_word;
}

inline bool sparse_matrix::get_row(int &cnt, row_element *curr_row, const Word_t &r) {
  //Word_t start_idx = r*col;
  //Word_t end_idx = (r+1)*col - 1;
  Pvoid_t PValue;
  Word_t Index = 0;
  JLF(PValue, Pmatrix[r], Index);
  cnt = 0;
  if (PValue == NULL) {
    //printf("row: %d Index: %d end_idx: %d\n", r, Index, end_idx);
    //printf("cnt : %d\n", cnt);
    return false;
  }
  curr_row[0].col = Index;
  curr_row[0].val = *(static_cast<Word_t*>(PValue));
  cnt = 1;
  while(1) {
    JLN(PValue, Pmatrix[r], Index);
    if (PValue != NULL) {
      curr_row[cnt].col = Index;
      curr_row[cnt].val = *(static_cast<Word_t*>(PValue));
      cnt++;
    } else {
      break;
    }
  }
  return true;
}
inline bool sparse_matrix::get_row_first(Word_t &val, Word_t &c, const Word_t &r) {
  //Word_t start_idx = r*col;
  //Word_t end_idx = (r+1)*col - 1;
  Pvoid_t PValue;
  Word_t Index = 0;
  JLF(PValue, Pmatrix[r], Index);
  if (PValue == NULL) {
    //printf("matrix row: %d row: %d Index: %d end_idx: %d\n", row, r, Index, end_idx);
    return false;
  } else {
    val = *(static_cast<Word_t*>(PValue));
    c = Index;
    return true;
  }
}
inline bool sparse_matrix::get_row_next(Word_t &val, Word_t &next_c, const Word_t &r, const Word_t &c) {
  Pvoid_t PValue;
  //Word_t end_idx = (r+1)*col - 1;
  Word_t Index = c;
  JLN(PValue, Pmatrix[r], Index);
  if (PValue == NULL) {
    return false;
  } else {
    val = *(static_cast<Word_t*>(PValue));
    next_c = Index;
    return true;
  }
}
#endif
