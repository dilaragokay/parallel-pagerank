#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <iterator>
#include <memory>
#include <math.h>
#include <metis.h>
#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unordered_map.hpp>

namespace mpi = boost::mpi;
using namespace std;

const string FILENAME = "/Users/dilaragokay/Parallel-PageRank/exampleGraph.txt";
const double epsilon = pow(10, -6);
const double alpha = 0.2;

vector<int> row_begin;
vector<double> values;
vector<int> col_indices;

vector<int> row_begin_metis;
vector<double> values_metis;
vector<int> col_indices_metis;

vector<double> first_5;
clock_t start; // start time
double duration; // how much time has passed during calculations

/**
 * @param rank Rank of the processor
 * @param part array which represents which node is assigned to which processor
 * @param M length of part array
 * @return vector of indices of nodes which are assigned to rank
 */
vector<int> findProcessors(int rank, idx_t *part, int M) {
    vector<int> indicesOfNodes;
    for (int i = 0 ; i < M ; i++) {
        if (part[i] == rank) {
            indicesOfNodes.push_back(i);
        }
    }
    return indicesOfNodes;
}

/**
 * Multiplies  P and r_t matrices in parallel
 *
 * parameters:
 * values_: the vector that contains values elements
 * col_indicies_: the vector that contains column indices elements
 * row_begin_: the vector that contains row begin elements
 * r_t_: the vector that contains r^t elements
 *
 */
double multiplication(vector<double> values_, vector<int> col_indices_, vector<int> row_begin_, vector<double> r_t_)
{
    double zi = 0;
    for (int it = row_begin_[0]; it < row_begin_[1]; it++)
    {
        int j = col_indices_[it];
        zi += values_[it] * r_t_[j];
    }
    return zi;
}


/**
 * Calculates Frobenius norm of difference of two vectors
 *
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

/**
 * Calculates r^(t+1) =αPr(t)+(1-α)c
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

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;

    cout << world.rank() << ", " << world.size() << '\n';
    if(world.rank() == 0){
        // MASTER
        int word_length = 26;  // word length of nodes

        // Read from file
        FILE *file;
        long size;
        char *buffer;
        size_t result;

        // Read the file as a binary
        file = fopen(FILENAME.c_str(), "rb");
        if (file == NULL) {
            fputs("File Error", stderr);
            exit(1);
        }

        // Go to end of the file
        fseek(file, 0, SEEK_END);

        // Find the size of the file
        size = ftell(file);
        // Go to beginning of the file
        rewind(file);
        buffer = (char *)malloc(sizeof(char) * size);

        if (buffer == NULL) {
            fputs("Memory Error", stderr);
            exit(2);
        }
        result = fread(buffer, 1, size, file);
        if (result != size) {
            fputs("Reading Error", stderr);
            exit(3);
        }
        cout << "File is created " << endl;
        string s;
        s.assign(&buffer[size - (word_length + 1)], word_length);

        unordered_map<string, int> input_list;  // name of the nodes and their indices
        // TODO: fix the comment for counter_list
        unordered_map<int, int> counter_list;  // unorderd map for counters of all nodes

        string old = "";
        int count = 0;

        string a, b;
        int index = 0;
        // Read all lines of the file and enumerate nodes that have incoming edge(s)
        for (int i = 0; i < size; i += ((word_length + 1) * 2)) {
            b.assign(&buffer[i + (word_length + 1)], word_length);
            unordered_map<string, int>::iterator it_b = input_list.find(b);
            if (it_b == input_list.end()) {
                input_list.insert(make_pair(b, index));
                counter_list.insert(make_pair(index, 0));
                index++;
            }
            if (old != b) {
                row_begin.push_back(count);
            }
            old = b;
            count++;
        }
        row_begin.push_back(count);
        // Read all lines and enumerate the nodes that don't have any incoming edge
        for (int i = 0; i < size; i += ((word_length + 1) * 2)) {
            a.assign(&buffer[i], word_length);
            unordered_map<string, int>::iterator it_a = input_list.find(a);

            if (it_a == input_list.end()) {
                input_list.insert(make_pair(a, index));
                counter_list.insert(make_pair(index, 1));
                row_begin.push_back(count);
                index++;
            }
            else {
                unordered_map<int, int>::iterator it_a_counter = counter_list.find(it_a->second);
                it_a_counter->second++;
            }

            it_a = input_list.find(a);

            col_indices.push_back(it_a->second);
        }
        // Fill values vector
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
         * counter_metis: counts the row number of the locations on the matrix.
         * row_begin_metis: new row vector for matrix.
         * col_indices_metis: new column vector for matrix.
         * values_metis: new values vector for matrix.
         *
         * row_counter_metis: counts the new elements that will be added to new row vector.
         * temp_col: stores elements that will be added to col_indices_metis.
         * --> temp_col must be sorted for CSR matrix format.
         */
        int counter_metis = 0;
        int row_counter_metis = 0;
        row_begin_metis.push_back(row_counter_metis);
        for (int i = 1; i < row_begin.size(); i++) {
            vector<int> temp_col;

            for (int j = row_begin[i-1]; j < row_begin[i]; j++) {
                temp_col.push_back(col_indices[j]);
            }
            for (int j = 0; j < col_indices.size(); j++) {
                if (col_indices[j] == counter_metis) {
                    for (int k = 0; k < row_begin.size(); k++) {
                        if (row_begin[k] == j) {
                            temp_col.push_back(k);
                            break;
                        }
                        else if (row_begin[k] > j) {
                            temp_col.push_back(k - 1);
                            break;
                        }
                    }
                }
            }

            sort(temp_col.begin(), temp_col.end());
            col_indices_metis.push_back(temp_col[0]);
            values_metis.push_back(1);

            for (int j = 1; j < temp_col.size(); j++) {
                col_indices_metis.push_back(temp_col[j]);
                values_metis.push_back(1);
            }

            counter_metis ++;
            row_counter_metis = col_indices_metis.size();
            row_begin_metis.push_back(row_counter_metis + 1);
        }
        idx_t nVertices = row_begin_metis.size() - 1;
        idx_t nParts = 3;
        idx_t balancingConstraint = 1;
        idx_t objval = 0;
        idx_t part[row_begin_metis.size()];

        idx_t options[METIS_NOPTIONS];
        METIS_SetDefaultOptions(options);
        options[METIS_OPTION_NUMBERING] = 1;
        options[METIS_OPTION_UFACTOR] = 15;
        options[METIS_OPTION_MINCONN] = 1;
        // Indexes of starting points in adjacent array
        idx_t xadj[row_begin_metis.size()];
        for (int l = 0; l < row_begin_metis.size(); l++) {
            xadj[l] = row_begin_metis[l];
        }
        // Adjacent vertices in consecutive index order
        idx_t adjncy[col_indices_metis.size()];

        for (int l = 0; l < col_indices_metis.size(); l++) {
            adjncy[l] = col_indices_metis[l];
        }
        int ret = METIS_PartGraphKway(
                &nVertices, // The number of vertices in the graph.
                &balancingConstraint,
                xadj,
                adjncy,
                NULL,
                NULL,
                NULL,
                &nParts, // The number of parts to partition the graph.
                NULL,
                NULL,
                options,
                &objval,
                part
                );
        cout << "ret " << ret << std::endl;
        for(unsigned part_i = 0; part_i < nVertices; part_i++){
            std::cout << part_i << " " << part[part_i] << std::endl;
        }

        ofstream myfile;
        myfile.open("output.csv");

        vector<double> r_t;
        vector<double> r_t_1(row_begin.size() - 1, 1.0);
        bool repeats = true;
        int counter = 0;
        do
        {
            r_t = r_t_1;
            // send P for multiplication and repeat
            for(int part_i = 1; part_i <= nParts; part_i++){
                vector<int> indicesOfNodes = findProcessors(part_i, part, nVertices);
                // send each to related processor
                world.send(part_i, 2, row_begin);
                world.send(part_i, 4, values);
                world.send(part_i, 5, col_indices);
                world.send(part_i, 0, indicesOfNodes);
                world.send(part_i, 1, r_t);
            }
            unordered_map<int, double> all_mult_results;
            for (int part_i = 1; part_i <= nParts; part_i++) {
                world.recv(part_i, 6, all_mult_results);
                for (auto& it: all_mult_results) {
                    r_t_1[it.first] = it.second;
                }
            }

            repeat(&r_t_1[0], M, alpha);
            counter++;
            double length = calculate_length(&r_t[0], &r_t_1[0], M);
            repeats = length > epsilon;
            world.send(1, 3, repeats);
            world.send(2, 3, repeats);
            world.send(3, 3, repeats);
        } while (repeats);
        first_5 = r_t_1;
        // TODO: return duration
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

        // print top 5 ranked strings.
        for (int k = 0; k < 5; k++) {
            for ( unordered_map<string, int>::iterator it_counter = input_list.begin(); it_counter != input_list.end(); ++it_counter ) {
                if (it_counter->second == arr_index[k]) {
                    cout << it_counter->first << endl;
                }
            }
        }
    } else {
        // SLAVE
        vector<int> indicesOfNodes;
        vector<double> r_t;
        bool repeat = true;
        vector<int> row_begin_received;
        vector<double> values_received;
        vector<int> col_indices_received;
        while (repeat) {
            world.recv(0, 2, row_begin_received);
            world.recv(0, 4, values_received);
            world.recv(0, 5, col_indices_received);
            world.recv(0, 0, indicesOfNodes);
            world.recv(0, 1, r_t);
            unordered_map<int, double> all_mult_results;
            for (int i = 0 ; i < indicesOfNodes.size() ; i++) {
                int index = indicesOfNodes[i];
                vector<int>::const_iterator first_int = row_begin_received.begin() + index;
                vector<int>::const_iterator last_int = row_begin_received.begin() + index + 2;
                vector<int> row_begin_slave(first_int, last_int);

                if (row_begin_slave[0] == row_begin_slave[1]) {
                    // values are finished
                    break;
                }
                double mult_result = multiplication(values_received, col_indices_received, row_begin_slave, r_t);
                all_mult_results.insert(make_pair(index, mult_result));
            }
            world.send(0, 6, all_mult_results);
            world.recv(0, 3, repeat);
        }
    }
    return 0;
}
