// compile with: nvcc -o a.out .\pagerank_thurst.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
// run with: .\a.out

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>
#include <ctime>
#include <memory>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <cmath>
#include <functional>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

using namespace std;

// CONSTANTS //
const string FILENAME = "graph.txt";
const double epsilon = pow(10, -6);
const double alpha = 0.2;
// CONSTANTS ENDS//

/*
  Returns absolute value of any type of value.
*/
template<typename T>
struct absolute_value : public unary_function<T,T>
{
  _host_ _device_ T operator()(const T &x) const
  {
    return x < T(0) ? -x : x;
  }
};

/*
  a: a constant
  x: a pointer
  Returns the result of  (a * x + (1 - a))
*/
struct saxpy_functor {
    const double a;
    saxpy_functor(double _a) : a(_a) {}

    _host_ _device_
        double operator()(const double& x) const {
            return a * x + (1 - a);
        }
};

void saxpy_fast(double A, thrust::device_vector<double>& X) {
    thrust::transform(X.begin(), X.end(), X.begin(), saxpy_functor(A));
}

int main() {
    ofstream myFile;
    myFile.open("output_thrust.csv");
    myFile << "Operation, Timing (s)" << endl;
    clock_t begin = clock();
    // main vectors of CSR format.
    thrust::host_vector<int> row_begin;     // row numbers of non-zero elements.
    thrust::host_vector<double> values;     // column numbers of non-zero elements.
    thrust::host_vector<int> col_indices;   // values of non-zero elements.

    // took word length of nodes.
    int word_length = 26;

    // read from file
    FILE *file;
    long size;
    char *buffer;
    size_t result;

    // read as a binary
    file = fopen(FILENAME.c_str(), "rb");
    if (file == NULL)
    {
        fputs("File Error", stderr);
        exit(1);
    }

    // go to end of the file
    fseek(file, 0, SEEK_END);

    // find the size of the file
    size = ftell(file);

    // go to start of the file
    rewind(file);

    buffer = (char *)malloc(sizeof(char) * size);
    if (buffer == NULL)
    {
        fputs("Memory Error", stderr);
        exit(2);
    }
    result = fread(buffer, 1, size, file);
    if (result != size)
    {
        fputs("Reading Error", stderr);
        exit(3);
    }

    cout << "file created " << endl;

    string s;
    s.assign(&buffer[size - (word_length + 1)], word_length);

    // umap of index for all Nodes
    unordered_map<string, int> input_list;
    // unorderd map for counters of all nodes.
    unordered_map<int, int> counter_list;

    string old = "";
    int count = 0;

    string a, b;
    int index = 0;

    /*
    read all lines and numbered only outgoing sides to edges.
    fill the row_begin vector.
    */
    for (int i = 0; i < size; i += ((word_length + 1) * 2))
    {
        b.assign(&buffer[i + (word_length + 1)], word_length);
        unordered_map<string, int>::iterator it_b = input_list.find(b);

        if (it_b == input_list.end())
        {
            input_list.insert(make_pair(b, index));
            counter_list.insert(make_pair(index, 0));
            index++;
        }
        if (old != b)
        {
            row_begin.push_back(count);
        }
        old = b;
        count++;
    }
    row_begin.push_back(count);

    /*
    read all lines and numbered necessery nodes.
    fill the col_indices vector.
    */
    for (int i = 0; i < size; i += ((word_length + 1) * 2))
    {
        a.assign(&buffer[i], word_length);
        unordered_map<string, int>::iterator it_a = input_list.find(a);

        if (it_a == input_list.end())
        {
            input_list.insert(make_pair(a, index));
            counter_list.insert(make_pair(index, 1));
            row_begin.push_back(count);
            index++;
        }
        else
        {
            unordered_map<int, int>::iterator it_a_counter = counter_list.find(it_a->second);
            it_a_counter->second++;
        }
        it_a = input_list.find(a);
        col_indices.push_back(it_a->second);
    }

    /*
    fill values vector.
    */
    int k = 0;
    for (int i = 1; i < row_begin.size(); i++)
    {
        for (int j = row_begin[i - 1]; j < row_begin[i]; j++)
        {
            unordered_map<int, int>::iterator it_counter = counter_list.find(col_indices[k]);
            if (it_counter != counter_list.end()) {
              values.push_back((double)1 / (it_counter->second));
            }
            k++;
        }
    }

    int M = row_begin.size() - 1;  // number of nodes
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    myFile << "I/O, " << elapsed_secs << endl;

    // initial vectors for multiplication
    thrust::host_vector<double> r_t_host(row_begin.size() - 1, 1.0);    // 1, 1, 1, 1, 1...
    thrust::host_vector<double> r_t_1_host(row_begin.size() - 1, 1.0);  // 1, 1, 1, 1, 1...

    // row_ctr will be multiply with values
    thrust::host_vector<int> row_ctr(values.size());
    for (int line = 1; line < row_begin.size(); line++) {
      for (int i = row_begin[line - 1]; i < row_begin[line]; i++) {
        row_ctr[i] = line;   // filled with row numbers
      }
    }

    double norm = pow(10, -6);    // initial value of norm is equal to epsilon
    begin = clock();
    // while L1 norm of r_t vector is larger than epsilon
    // Matrix multiplication is done in two parts
    // 1. multiplying corresponding elements one by one
    // 2. summing the multiplied elements calculated in 1
    while (norm >= epsilon) {
      thrust::device_vector<double> r_t = r_t_host;
      thrust::device_vector<double> r_t_1 = r_t_1_host;

      r_t = r_t_1;
      thrust::host_vector<double> r_t_ = r_t;
      thrust::host_vector<double> mult_vec(col_indices.size());

      // Write values of r_t which correspond to a column
      for (int i = 0; i < col_indices.size(); i++) {
        mult_vec[i] = r_t_[col_indices[i]];
      }

      thrust::device_vector<double> mult_vec_new = mult_vec;
      thrust::device_vector<double> reisss(values.size());
      thrust::multiplies<double> multOp;

      thrust::device_vector<double> val_device = values;
      // Multiply val_device with mult_vec_new elementwise and write the result to reisss
      thrust::transform(val_device.begin(), val_device.end(), mult_vec_new.begin(), reisss.begin(), multOp );

      thrust::device_vector<int> row_device = row_begin;
      thrust::device_vector<int> row_ctr_new = row_ctr;

      // row_ctr_new has the same size of values. For each value in values, the
      // element in the row_ctr_new with the same index is the row number of the
      // value. reduce_by_key sums the values in reisss which are in the same
      // row. Namely, it reducesrow_ctr_new to new r_t_1 vector which has the
      // final matrix multiplication results
      thrust::reduce_by_key(row_ctr_new.begin(), row_ctr_new.end(), reisss.begin(), row_device.begin(), r_t_1.begin());

      saxpy_fast(alpha, r_t_1);

      thrust::minus<double> op2;
      thrust::transform(r_t_1.begin(), r_t_1.end(), r_t.begin(), r_t.begin(), op2 );

      thrust::transform(r_t.begin(), r_t.end(), r_t.begin(), absolute_value<double>());

      norm = thrust::reduce(r_t.begin(), r_t.end());

      r_t_host = r_t;
      r_t_1_host = r_t_1;
    }
    end = clock();
    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    myFile << "PageRank, " << elapsed_secs << endl;

    thrust::host_vector<double> first_5 = r_t_1_host;
    // took maximum 5 ranks and second array took the indexes of them.
    int arr[5] = {0, 0, 0, 0, 0};
    int arr_index[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < first_5.size(); i++) {
        if (first_5[i] > arr[0]) {
            arr[4] = arr[3];
            arr_index[4] = arr_index[3];
            arr[3] = arr[2];
            arr_index[3] = arr_index[2];
            arr[2] = arr[1];
            arr_index[2] = arr_index[1];
            arr[1] = arr[0];
            arr_index[1] = arr_index[0];
            arr[0] = first_5[i];
            arr_index[0] = i;
        } else if (first_5[i] > arr[1]) {
            arr[4] = arr[3];
            arr_index[4] = arr_index[3];
            arr[3] = arr[2];
            arr_index[3] = arr_index[2];
            arr[2] = arr[1];
            arr_index[2] = arr_index[1];
            arr[1] = first_5[i];
            arr_index[1] = i;
        } else if (first_5[i] > arr[2]) {
            arr[4] = arr[3];
            arr_index[4] = arr_index[3];
            arr[3] = arr[2];
            arr_index[3] = arr_index[2];
            arr[2] = first_5[i];
            arr_index[2] = i;
        } else if (first_5[i] > arr[3]) {
            arr[4] = arr[3];
            arr_index[4] = arr_index[3];
            arr[3] = first_5[i];
            arr_index[3] = i;
        } else if (first_5[i] > arr[4]) {
            arr[4] = first_5[i];
            arr_index[4] = i;
        }
    }
    myFile << "Top 5 hosts," << endl;
    // print top 5 ranked strings.
    for (int k = 0; k < 5; k++) {
        for ( unordered_map<string, int>::iterator it_counter = input_list.begin(); it_counter != input_list.end(); ++it_counter ) {
            if (it_counter->second == arr_index[k]) {
                myFile << it_counter->first << "," << endl;
            }
        }
    }
    myFile.close();
    return 0;
}
