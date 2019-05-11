#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>
#include <memory>
#include <math.h>
#include <omp.h>

using namespace std;

// CONSTANTS //

const string FILENAME = "graph.txt";
const double epsilon = pow(10, -6);
const double alpha = 0.2;

// CONSTANTS ENDS//

vector<int> row_begin;
vector<double> values;
vector<int> col_indices;
vector<double> first_5;
clock_t start; // to start time
double duration; // to calculate how much time has passed during calculations

/* to multiply  P and r_t matrices paralelly
 * The schedule(static, chunk-size) clause of the loop construct
 * specifies that the for loop has the static scheduling type.
 *
 * parameters:
 * r^t_1: output vector for multiplication (P*r^t)
 * values_: the vector that contains values elements
 * col_indicies_: the vector that contains column indices elements
 * row_begin_: the vector that contains row begin elements
 * r_t_: the vector that contains r^t elements
 * M: length of original matrix
 * chunk: chunk size
 * num_threads: number of threads
 */
void static_multiplication(double *r_t_1_, double *values_, int *col_indices_, int *row_begin_, double *r_t_, int M, int chunk, int num_threads)
{
    long i, j, it;
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for private(it, j) schedule(static, chunk)
        for (i = 0; i < M; i++)
        {
            double zi = 0.0;
            for (it = row_begin_[i]; it < row_begin_[i + 1]; it++)
            {
                j = col_indices_[it];
                zi += values_[it] * r_t_[j];
            }
            r_t_1_[i] = zi;
        }
    }
}

/* to multiply  P and r_t matrices paralelly
 * The schedule(dynamic, chunk-size) clause of the loop construct
 * specifies that the for loop has the dynamic scheduling type.
 *
 * parameters:
 * r^t_1: output vector for multiplication (P*r^t)
 * values_: the vector that contains values elements
 * col_indicies_: the vector that contains column indices elements
 * row_begin_: the vector that contains row begin elements
 * r_t_: the vector that contains r^t elements
 * M: length of original matrix
 * chunk: chunk size
 * num_threads: number of threads
 */
void dynamic_multiplication(double *r_t_1_, double *values_, int *col_indices_, int *row_begin_, double *r_t_, int M, int chunk, int num_threads)
{
    long i, j, it;
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for private(it, j) schedule(dynamic, chunk)
        for (i = 0; i < M; i++)
        {
            double zi = 0.0;
            for (it = row_begin_[i]; it < row_begin_[i + 1]; it++)
            {
                j = col_indices_[it];
                zi += values_[it] * r_t_[j];
            }
            r_t_1_[i] = zi;
        }
    }
}

/* to multiply  P and r_t matrices paralelly
 * The schedule(guided, chunk-size) clause of the loop construct
 * specifies that the for loop has the guided scheduling type.
 *
 * parameters:
 * r^t_1: output vector for multiplication (P*r^t)
 * values_: the vector that contains values elements
 * col_indicies_: the vector that contains column indices elements
 * row_begin_: the vector that contains row begin elements
 * r_t_: the vector that contains r^t elements
 * M: length of original matrix
 * chunk: chunk size
 * num_threads: number of threads
 */
void guided_multiplication(double *r_t_1_, double *values_, int *col_indices_, int *row_begin_, double *r_t_, int M, int chunk, int num_threads)
{
    long i, j, it;
#pragma omp parallel num_threads(num_threads)
    {
#pragma omp for private(it, j) schedule(guided, chunk)
        for (i = 0; i < M; i++)
        {
            double zi = 0.0;
            for (it = row_begin_[i]; it < row_begin_[i + 1]; it++)
            {
                j = col_indices_[it];
                zi += values_[it] * r_t_[j];
            }
            r_t_1_[i] = zi;
        }
    }
}

/* to calculate length
 * if the sum of the absolute values of each element of these two vectors is
 * below a certain threshold (𝜀), iteration ends. the method is for calculate this value
 *
 * parameters:
 * r^t_1: output vector for multiplication (α*P*r^t+(1-α)*c)
 * r_t_: the vector that contains r^t elements
 * M: length of original matrix
 *
 * return value:
 * ||r^(t+1) − r^(t)||
 */
double calculate_length(double *r_t_, double *r_t_1_, int M)
{
    double sum = 0;
    for (int i = 0; i < M; i++)
    {
        sum += abs(r_t_1_[i] - r_t_[i]);
    }
    return sum;
}

/* calculates r^(t+1) =αPr(t)+(1-α)c
 *
 * parameters:
 * r^t_1: first, P*r^(t). After calculation: αPr^(t)+(1-α)c
 * M: length of original matrix
 * alpha: value of α
 */
void repeat(double *r_t_1_, int M, double alpha)
{
    for (int i = 0; i < M; i++)
    {
        r_t_1_[i] = r_t_1_[i] * alpha + (1 - alpha);
    }
}

/* does loop body
 *
 * parameters:
 * M: length of original matrix
 * chunk: chunk size
 * num_threads: number of threads
 * schedule_type: schedule type
 *     Static ---> 1
 *     Dynamic ---> 2
 *     Guided ---> 3
 *
 * return value: string contains timings for each number of threads (in secs)
 */
string body_loop(int M, int chunk, int num_threads, int schedule_type)
{

    vector<double> r_t;
    vector<double> r_t_1(row_begin.size() - 1, 1.0);

    int counter = 0;
    start = clock();
    do
    {
        r_t = r_t_1;
        if (schedule_type == 1)
        {
            static_multiplication(&r_t_1[0], &values[0], &col_indices[0], &row_begin[0], &r_t[0], M, chunk, num_threads);
        }
        else if (schedule_type == 2)
        {
            dynamic_multiplication(&r_t_1[0], &values[0], &col_indices[0], &row_begin[0], &r_t[0], M, chunk, num_threads);
        }
        else if (schedule_type == 3)
        {
            guided_multiplication(&r_t_1[0], &values[0], &col_indices[0], &row_begin[0], &r_t[0], M, chunk, num_threads);
        }
        repeat(&r_t_1[0], M, alpha);

        counter++;

    } while (calculate_length(&r_t[0], &r_t_1[0], M) > epsilon);
    //cout << schedule_type << " - Iteration number: " << counter << endl;
    duration = (omp_get_wtime() - start) / (double)CLOCKS_PER_SEC;
    //cout << "Finished: " << duration << endl;
    first_5 = r_t_1;
    if (num_threads == 1)
    {
        return (to_string(counter) + ", " + to_string(duration));
    }
    return to_string(duration);
}

/* was created for ease of timing calculations
 *
 * parameters:
 * M: length of original matrix
 * chunk: chunk size
 * schedule_type: schedule type
 *     Static ---> 1
 *     Dynamic ---> 2
 *     Guided ---> 3
 *
 * return value: string contains timings for all number of threads (in secs)
*/
string body_main(int M, int chunk, int schedule_type)
{
    string th_1 = body_loop(M, chunk, 1, schedule_type);
    string th_2 = body_loop(M, chunk, 2, schedule_type);
    string th_4 = body_loop(M, chunk, 4, schedule_type);
    string th_8 = body_loop(M, chunk, 8, schedule_type);
    string r = th_1 + ", " + th_2 + ", " + th_4 + ", " + th_8;
    return r;
}

int main()
{
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

    int M = row_begin.size() - 1; // number of nodes

    /*
    Static ---> 1,
    Dynamic ---> 2,
    Guided ---> 3,
    */

    /* creates csv file and writes into.
     * This file contains:
     *    -Test No.
     *    -Scheduling Method
     *    -Chunk Size
     *    -No. of Iterations
     *    -Timings in secs for each number of threads
     */
    ofstream myfile;
    myfile.open("output.csv");
    myfile << "Test No., Scheduling Method, Chunk Size, No. of Iterations, 1, 2, 4, 8" << endl;
    myfile << "1, static, "<< "500, "<< body_main(M, 500, 1) << endl;  // M, chunk, scheduler_type
    myfile << "2, static, "<< "1000, "<< body_main(M, 1000, 1) << endl; // M, chunk, scheduler_type
    myfile << "3, static, "<< "1500, "<< body_main(M, 1500, 1) << endl; // M, chunk, scheduler_type
    myfile << "4, dynamic, "<< "500, "<< body_main(M, 500, 2) << endl;  // M, chunk, scheduler_type
    myfile << "5, dynamic, "<< "1000, "<< body_main(M, 1000, 2) << endl; // M, chunk, scheduler_type
    myfile << "6, dynamic, "<< "1500, "<< body_main(M, 1500, 2) << endl; // M, chunk, scheduler_type
    myfile << "7, guided, "<< "500, "<< body_main(M, 500, 3) << endl;  // M, chunk, scheduler_type
    myfile << "8, guided, "<< "1000, "<< body_main(M, 1000, 3) << endl; // M, chunk, scheduler_type
    myfile << "9, guided, "<< "1500, "<< body_main(M, 1500, 3) << endl; // M, chunk, scheduler_type
    myfile.close();

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

    // print top 5 ranks strings.
    for (int k = 0; k < 5; k++) {
      for ( unordered_map<string, int>::iterator it_counter = input_list.begin(); it_counter != input_list.end(); ++it_counter ) {
        if (it_counter->second == arr_index[k]) {
          cout << it_counter->first << endl;
        }
      }
    }

    return 0;
}
