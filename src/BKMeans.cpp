/*--------------------------------------
 BKMeans for compact binary code

    Compile option -I../include/libbkmeans -msse4.2 -fopenmp'
    use environment variable OMP_NUM_THREADS to specify thread numbers
-------------------------------------*/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>
#include <BKMeans.h>

using namespace std;

BKMeans::~BKMeans()
{
    free(_centroids);
    free(_newcentroids);
    free(_counts);
    free(_bitsums);
    _assignment.clear();
}

int BKMeans::findNearestCenter(uint64_t *s)
{
    int id = 0, mindist = 9999999, dist;
    if (_centroids == NULL || _samples == NULL) {
        cerr << "ERROR: centroids or samples are empty" << endl;
        return -1; 
    }

    // Compute distance with all centroids
    for (int i = 0; i < _k; i++) {
        dist = hamdist(&_centroids[i*_width], s);
        if (dist < mindist) {
            mindist = dist;
            id = i;
        }
    }

    if (verbose > 2)
        printf("BKMeans::findNearestCenter: %d %d\n", id, mindist);

    return id;
}

std::vector<unsigned int> BKMeans::findAllNearestCenters(uint64_t *s)
{
    int id = 0, mindist = 9999999, dist;
    std::vector<unsigned int> v;

    // Compute distance with all centroids
    for (int i = 0; i < _k; i++) {
        dist = hamdist(&_centroids[i*_width], s);
        if (dist < mindist) {
            mindist = dist;
            v.clear();
            v.push_back(i);
        } else if (dist == mindist) {
            v.push_back(i);
        }
    }

    return v;
}

/* centroid and distance */
typedef struct centdist {
    int centroid;
    int dist;
} centdist;    

bool comparecd(centdist a, centdist b)
{
    return a.dist < b.dist;
}

std::vector<unsigned int> BKMeans::findAllNearestCentersThreshold(uint64_t *s)
{
    int mindist = 9999999, dist, id = 0;
    int *distances = new int[_k];
    centdist *cd = new centdist[_k];
    std::vector<unsigned int> v;

    // Compute distance with all centroids, find the minimum
    for (int i = 0; i < _k; i++) {
        dist = hamdist(&_centroids[i*_width], s);
        if (verbose) 
            printf(" %d", dist);
        if (dist < mindist) {
            mindist = dist;
            id = i;
        }
        distances[i] = dist;
    }
    _counts[id]++;
    if (verbose) 
       printf("minid=%d mindist: %d\n", id, mindist);

    // all the centroids which distance <= threshold
    int num = 0;
    for (int i = 0; i < _k; i++) {
      if (distances[i] <= mindist/_threshold) {
	    cd[num].dist = distances[i];
	    cd[num].centroid = i;
	    num++;
	  }
    }

    std::sort(cd, cd+num, comparecd);
    for (int i=0; i<num; i++) {
        v.push_back(cd[i].centroid);
    }

    delete []distances;
    delete []cd;
    return v;
}

std::vector<unsigned int> BKMeans::findAllNearestCentersDisthres(uint64_t *s, int disthres)
{
    int mindist = 9999999, dist;
    int *distances = new int[_k];
    centdist *cd = new centdist[_k];
    std::vector<unsigned int> v;

    // Compute distance with all centroids, find the minimum
    for (int i = 0; i < _k; i++) {
        dist = hamdist(&_centroids[i*_width], s);
        if (dist < mindist) {
            mindist = dist;
        }
        distances[i] = dist;
    }

    // all the centroids which distance <= threshold
    int num = 0;
    for (int i = 0; i < _k; i++) {
        if (distances[i] <= mindist+disthres) {
	    cd[num].dist = distances[i];
	    cd[num].centroid = i;
	    num++;
	}
    }

    std::sort(cd, cd+num, comparecd);
    for (int i=0; i<num; i++) {
        v.push_back(cd[i].centroid);
    }

    delete []distances;
    delete []cd;
    return v;
}

/* initialize the centroids randomly */
int BKMeans::initRandCentroids()
{
    srand(time(NULL));

    std::vector<uint32_t> tmp(_n);
    for (int i = 0; i < _n; i++) {
        tmp[i] = i;
    }
    // Shuffle   
    std::random_shuffle (tmp.begin(), tmp.end());

    if (verbose > 0)
        printf("init centroids: ");
    // Take K firsts
    for (int i = 0; i < _k; i++) {
        memcpy(_centroids+i*_width, _samples+tmp[i]*_width, _width*sizeof(uint64_t));
        if (verbose > 0)
            printf("%d ", tmp[i]);
    }
    if (verbose > 0)
        printf("\n");
    tmp.clear();

    if (verbose > 0) {
        for (int i = 0; i < _k; i++) {
            printf("centroids[%d]: ", i);
            for (int w=0; w<_width; w++) {
                printf("%16lx ", _centroids[i*_width + w]);
            }
            printf("\n");
        }
    }

    // OK
    return 0;
}

/* trainning function */
int BKMeans::cluster()
{
    bool stop;
    int iter = 0, retcode = 0;
    int bestcenter;

    int i,j,w;

    time_t begintime, endtime;
    time(&begintime);
    // Init: choose initial centroids
    int iret = initRandCentroids();
    if (iret != 0) {
        cerr << "ERROR: cannot init centroids" << endl;
        return -1;
    }

#if 0
    for (i=0; i<_k*_width*2; i++) {
        *(int *)((char *)_centroids+ i*4) = rand();
    }
#endif

    stop = false;
    iter = 0;
    while (!stop) {
        memset(_counts, 0, _k*sizeof(int));
        memset(_bitsums, 0, _k*_bit_width*sizeof(int));
        // Assign samples to centroids and compute bitsums
        
        #pragma omp parallel for private(bestcenter, w, j, i) 
        for (i = 0; i < _n; i++) {
            bestcenter = findNearestCenter(_samples + i*_width);
            #pragma omp atomic
            _counts[bestcenter]++;
            
            if (verbose > 2) {
                printf("sample %d bestcenter=%d counts:%d\n", i+1, bestcenter, _counts[bestcenter]);
            }

            for (w=0; w<_width; w++) {
                for (j=0; j<64; j++) {
                    if (_samples[i*_width+w] & (1L<<j)) {
                        #pragma omp atomic
                        _bitsums[bestcenter*_bit_width + w*64 +j]++;
                    }
                }
            }
        }

        if (verbose > 1) {
            for (i=0; i<_k; i++) {
                printf("bitsum[%d]: ", i);
                for (j=0; j<_bit_width; j++) {
                    printf("%d ", _bitsums[i*_bit_width+j]);
                }
                printf("\n");
            }
        }

        // Find new centroids
        int updates = 0;
        memset(_newcentroids, 0, _k*_width*sizeof(uint64_t));

        int bitsum2;
        #pragma omp parallel for private(bitsum2, i, w, j) 
        for (int i = 0; i < _k; i++) {
            if (verbose > 0) 
                printf("centers[%d] counts:%d\n", i, _counts[i]);
            if (_counts[i] == 0) {
                memset(&_centroids[i*_width], 0, _width*sizeof(uint64_t));
                continue;
            }
            for (int w=0; w<_width; w++) {
                for (int j=0; j<64; j++) {
                    bitsum2 = _bitsums[i*_bit_width + w*64 +j] * 2;
                    if (bitsum2 >= _counts[i]) {
                        #pragma omp atomic
                        _newcentroids[i*_width +w] |= 1L<<j;
                    }
                }
            }
            if (hamdist(&_newcentroids[i*_width], &_centroids[i*_width]) > epsilon) {
                if (verbose)
                    printf("--- centroid %d changed ---\n", i);
                updates++;
            }
            memcpy(&_centroids[i*_width], &_newcentroids[i*_width], _width*sizeof(uint64_t));
        }

        if (verbose > 0) {
            for (int i = 0; i < _k; i++) {
                printf("new centroids[%d]: ", i);
                for (int w=0; w<_width; w++) {
                    printf("%16lx ", _newcentroids[i*_width + w]);
                }
                printf("\n");
            }
        }

        if (verbose)
            allCentroidsDistances();

        //mergeCentroids();

        wssestruct w = computeCost();
        printf("----------- Iteration: %d, WSSSE: %f %f ----------\n", iter+1, w.cost1, w.cost2);
        // Evaluate new cost and check stop condition
        if (updates == 0) {
            stop = true;
            printf("\nINFO: algorithm fully converged in %d iterations\n", iter+1);
            retcode = BKMEANS_RET_CONV;
        }

        // New iteration finished
        iter++;
        if (iter >= _maxIters) {
            stop = true;
            cout << endl << "INFO: max iterations reached" << endl;
            retcode = BKMEANS_RET_MAXIT;
        }
    }
    time(&endtime);
    printf("--------- Train finished in %ld seconds -----------\n", endtime+1-begintime);

    return retcode;
}

std::vector<uint64_t *> BKMeans::get_centroids()
{
        std::vector<uint64_t *> centroids;

        for (int i = 0; i < _k; i++) {
            centroids.push_back(_centroids+i*_width);
        }

        return centroids;
}

uint64_t  *BKMeans::get_centroids_pointer()
{
    return _centroids;
}

std::vector< std::vector<unsigned int> > BKMeans::calc_assignment()
{
    if (assignFlag)
        return _assignment;

    memset(_counts, 0, _k*sizeof(int));
    std::vector<unsigned int> v;
    for (int i = 0; i < _n; i++) {
        if (verbose)
            printf("i=%d ", i+1);
        _assignment.push_back(findAllNearestCentersThreshold(_samples + i*_width));
    }
    
    if (verbose) {
        for (int i=0; i<_k; i++) {
            printf("i=%d, count=%d\n", i, _counts[i]);
        }
    }

    assignFlag = true;
    return _assignment;
}

void BKMeans::allDistances()
{
    int i, j;

    for (i=0; i<_n; i++) {
        for (j=0; j<_n; j++) {
            printf("%d ", hamdist(_samples+i*_width, _samples+j*_width));
        }
        printf("\n");
    }
}

void BKMeans::allCentroidsDistances()
{
    int i, j, dist;

    for (i=0; i<_k; i++) {
        for (j=0; j<_k; j++) {
            dist = hamdist(_centroids+i*_width, _centroids+j*_width);
            printf("%4d ", dist);
        }
        printf("\n");
    }
}


void BKMeans::mergeCentroids()
{
    int i, j, dist;

    for (i=0; i<_k; i++) {
        for (j=0; j<_k; j++) {
            dist = hamdist(_centroids+i*_width, _centroids+j*_width);
            if (j != i && dist < 10) {
                printf("--- two centroid %d %d dist(%d) too close, converge to one centroid\n", j ,i, dist);
                memset(_centroids+j*_width, 0, _width*8);
            }
        }
    }
}

/* compute WSSSE(Within Set Sum of Squared Errors) */
wssestruct BKMeans::computeCost()
{
    double cost;
    wssestruct wsse;
    wsse.cost1 = 0;
    wsse.cost2 = 0;

    int bestcenter, i;
    #pragma omp parallel for private(bestcenter, cost) 
    for (int i=0; i<_n; i++) {
        bestcenter = findNearestCenter(_samples + i*_width);
        cost = hamdist(&_samples[i*_width], &_centroids[bestcenter*_width]);
        #pragma omp atomic
        wsse.cost1 += cost;
        #pragma omp atomic
        wsse.cost2 += cost*cost;
    }

    wsse.cost2 = sqrt(wsse.cost2);
    return wsse;
}


/* save trained model (centroids) to file */
int BKMeans::saveModel(const char *modelfile)
{
    FILE *f = fopen(modelfile, "w");
    if (f == NULL) {
        printf("Model file %s can't be opened for write: %s\n", modelfile, strerror(errno));
        return -1;
    }

    for (int i=0; i<_k; i++) {
        for (int w=0; w<_width; w++) {
            fprintf(f, "%16lx ", _centroids[i*_width + w]);
        }
        fprintf(f, "\n");
    } 

    fclose(f);
    return 0;
}

/* load trained model (centroids) from file */
int BKMeans::loadModel(const char *modelfile)
{
    ifstream fin(modelfile);
    if (!fin) { std::cerr << "Error opening model file!\n"; return -1; }

    string s;
    int n=0;
    while (fin >> s) {
        _centroids[n++] = strtoull(s.c_str(), NULL, 16);
        if (n > _k * _width) {
            _k *= 2;
            _centroids = (uint64_t *)realloc(_centroids, _k * _width * sizeof(uint64_t));
        }
    }
    if (_k > n/_width)
        _k = n/_width;

    printf("k=%d \n", _k);
        
    return 0;
}

