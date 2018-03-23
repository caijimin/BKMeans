/* g++ BKMeans.cpp test_BKMeans.cpp -o test_BKMeans -I../include/libbkmeans -msse4.2 */
/* To add openmp support, please specify '-fopenmp' compile option */

#include <stdio.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <string.h>
#include <stdint.h>
#include <nmmintrin.h>
#include <unistd.h>
#include <stdlib.h>
using namespace std;

#include "BKMeans.h"

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

string str1 = "18315646214247482355 18374965815763764168 17567425784458739964 16424118698164366140 14907100501358161835 45814174470351596 9816650587782916834 2538825140866687656 3606449415323658240 10272300289317577258 9992005062842393272 13675858221185874778 5601020766420218824 1035114119630512345 11078150460601385581 1535332807022184787";
string str2 = "18320149813874885619 18086737088883966668 12955739766022981884 14118416426439032626 14979232708608131631 11576505053999216828 10104739048854823658 190200122404021228 3498377310385086466 12663711132971364411 783832709123287208 11649344230033635942 7202051396089782681 14950117425651353225 5239227893432566513 6134810655804881503";

void usage(char *name)
{
    printf("usage: %s [B] [-b bit_width] [-k cluster_number] [-i iteration] [-f datafile] [-v] [-h]\n\n", name);

    printf("\t-B datafile byte format like '184 81 173 232' instead of defalut uint64_t format '2867679328082739030'\n");
    printf("\t-b bit_width, default 1024\n");
    printf("\t-k the number of clusters (centroids), default 20\n");
    printf("\t-i iterations, default 50\n");
    printf("\t-f datafile name, default ../data/binary_codes_6000\n");
    printf("\t-v verbose\n");
    printf("\t-c computeCost\n");
    printf("\t-h help\n");
}

void printBits(uint64_t *p)
{
    int i;
    for (i=0; i<64; i++) {
        printf("%u", *p & (1L<<i) ? 1:0 );
        if (i%8==7) 
            printf(" ");
    }
}


int main(int argc, char **argv)
{
    long i;
    std::vector<string> dbcodes_vec1 = split(str1, ' ');
    if (dbcodes_vec1.size() != 16)
        cout<< dbcodes_vec1.size();
    std::vector<string> dbcodes_vec2 = split(str2, ' ');
    if (dbcodes_vec2.size() != 16)
        cout << dbcodes_vec2.size();
    int bit_width = 1024;
    int n = 0, K=20, iter = 50;
    int verbose = 0;
    int computeCost = 0;
    char datafile[512];
    int byteFormat = 0;
    strcpy(datafile,  "../data/binary_codes_6000");

    int c;
    while ((c = getopt (argc, argv, "Bc:b:k:i:f:vh")) != -1) {
        switch (c)
        {
            case 'B':
                byteFormat = 1;
                break;
            case 'c':
                computeCost = 1;
                break;
            case 'b':
                bit_width = atoi(optarg);
                break;
            case 'k':
                K = atoi(optarg);
                break;
            case 'i':
                iter = atoi(optarg);
                break;
            case 'f':
                strcpy(datafile, optarg);
                break;
            case 'v':
                verbose++;
                break;
            case 'h':
                usage(argv[0]);
                return 0;
            default:
                usage(argv[0]);
                return 0;
        }

    }

    int num_bit_64 = bit_width/64;
    char line[512];
    
    uint64_t maxsize = 102*1024;
    uint64_t *data = (uint64_t *)malloc(maxsize *sizeof(uint64_t));
    /* read data from file */
    ifstream fin(datafile);
    if (!fin) { std::cerr << "Error opening data file!\n"; return -1; }

    string s;
    i = 0;
    while (fin >> s) {
        if (byteFormat) {
            *((unsigned char *)data + i) = atoi(s.c_str());
            i++;
            if (i %8 == 0)
                n++;
        } else {
            data[n++] = strtoull(s.c_str(), NULL, 0);
        }
        if (n >= maxsize) {
            maxsize *= 2;
            data = (uint64_t *)realloc(data, maxsize *sizeof(uint64_t));
            if (data == NULL) {
                printf("Allocate memory error\n");
                exit(1);
            }
        }
    }
   /* uint64_t x;
    printf("\n----\n");
    for(i=32; i<48;i ++) {
        x = data[i] ^ data[i+16];
        printBits(&x);
    }
    printf("\n----\n");*/
    /*
    printf("n=%d\n", n);
            printf("%16lx ", data[0]);
            printf("%16lx ", data[95999]);*/
    fin.close();
    n /= num_bit_64;
    printf("n=%d\n", n);

/*    for (int j = 0; j < num_bit_64; j++)
        data[j] = strtoull(dbcodes_vec1[j].c_str(), NULL, 0);
    for (int j = 0; j < num_bit_64; j++)
        data[num_bit_64+j] = strtoull(dbcodes_vec2[j].c_str(), NULL, 0);
    data[0] = 0x1234567812345678L;
    data[1] = 0x2234567812345678L;
    data[2] = 0x3234567812345678L;
    data[3] = 0x2234567812345678L;
    data[4] = 0xb234567812345678L;
    data[5] = 0x2234567812345678L;
    data[6] = 0x3334567812345678L;
    data[7] = 0x3234567812345678L;
*/

    printf("------------------ begin --------------\n");
    BKMeans bkm(data, n, bit_width, K, iter, 1);
    if (verbose)
        bkm.setverbose(verbose);
    if (verbose)
        bkm.allDistances();

    bkm.cluster();
    /* save model to file or load model from file */
    //bkm.saveModel("model_file");
    //bkm.loadModel("model_file");
   // uint64_t *cp = bkm.get_centroids_pointer();

    return 0;
    printf("[INFO] print all assignments\n");
    std::vector< std::vector<unsigned int> > assign = bkm.calc_assignment();
    i=1;
    for (std::vector< std::vector<unsigned int> >::iterator it= assign.begin(); it != assign.end(); ++it) {
        printf("%d ", i);
        for (std::vector<unsigned int>::iterator it2= (*it).begin(); it2 != (*it).end(); ++it2) {
            printf("%d ", *it2);
        }
        printf("\n");
        i++;
    }
    if (verbose)
        bkm.allCentroidsDistances();

    free(data);
    return 0;
}

