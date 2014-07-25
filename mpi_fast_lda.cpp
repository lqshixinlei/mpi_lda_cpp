#include "mpi_fast_lda.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  lda model(100, 0.05, 0.05, 500);
  string part, train_file;
  char buf[10];
  sprintf(buf, "%05d", model.rank);
  part = buf;
  train_file = "train_data_"+part;
  cout << train_file << endl;
  model.read_words_dict("./words_dict");
  cout << "words_dict read complete!" << endl;
  model.read_trn_data(train_file);
  cout << "train data read complete!" << endl; 
  model.init_est();
  model.estimate();
  if (model.rank == 0) {
    model.compute_phi();
    model.save_model_twords();
  }
  //test();
  MPI_Finalize();
  return 0;
}
