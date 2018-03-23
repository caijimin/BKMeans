//--------------------------------------
// BKMeans for compact binary code
//--------------------------------------

#ifndef _BKMEANS_H_
#define _BKMEANS_H_ 1

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <stdint.h>
#include <nmmintrin.h>
#include <algorithm>

#define BKMEANS_RET_CONV  1
#define BKMEANS_RET_MAXIT 2

typedef struct wssestruct {
    double cost1;
    double cost2;
} wssestruct;

/*
   class BKMeans
   This class implements kmeans for binary data.
*/

class BKMeans
{
    public:      
        BKMeans(uint64_t *binary_data, size_t n, size_t bit_width, size_t K, size_t iter_max, 
                float t_for_assign = 1) { 

            epsilon = 0;
            _samples = binary_data; 
            _maxIters = iter_max;
            _k = K;
            _n = n;
            if ((bit_width % 64) != 0) {
                printf("Error: bit_width %zu is not multiple of 64\n", bit_width);
            } else {
                _width = bit_width/64;
            }
            _bit_width = bit_width;
            _threshold = t_for_assign; 
            if (_threshold == 0 ) {
                printf("Threshold can't be 0, use default value 1.0\n");
                _threshold = 1.0;
            }
            _centroids = (uint64_t *)malloc(K * _width * sizeof(uint64_t));
            _newcentroids = (uint64_t *)malloc(K * _width * sizeof(uint64_t));
            _counts = (int *)calloc(K, sizeof(int));
            _bitsums =(int *)calloc(K*bit_width, sizeof(int));
            if (_centroids == NULL || _newcentroids == NULL || _counts == NULL || _bitsums == NULL) {
                printf("Allocate memory error\n");
                exit(1);
            }
            assignFlag = false;
            verbose = 0;
        };

        /* Train the data.
            Returns:
               0: normal
              -1: exception 
        */
        int cluster();

        /* compute WSSSE(Within Set Sum of Squared Errors) */
        wssestruct computeCost();

        /* Calculate the assignment for every point, a point may belong to multiple clusters */
        std::vector< std::vector<unsigned int> > calc_assignment();

        ~BKMeans();

	    /* Returns a copy of the centroids */
        std::vector<uint64_t *> get_centroids();

        uint64_t *get_centroids_pointer();

        /* find nearest center index */
        int findNearestCenter(uint64_t *s);

        /* find all nearest centers with same distance */
        std::vector<unsigned int> findAllNearestCenters(uint64_t *s);

        /* find nearest center first, then find all centers which distance difference 
           is less than or equal to _threshold */
        std::vector<unsigned int> findAllNearestCentersThreshold(uint64_t *s);

        /* find nearest center first, then find all centers which distance difference 
           is less than or equal to disthres */
        std::vector<unsigned int> findAllNearestCentersDisthres(uint64_t *s, int disthres);

        void setverbose(int level) { verbose = level;};

        /* save trained model (centroids) to file */
        int saveModel(const char *modelfile);
        /* load trained model (centroids) from file */
        int loadModel(const char *modelfile);

        /* printf all distances */
        void allDistances();
        void allCentroidsDistances();
        void mergeCentroids();

        /* Computes Hamming distance between two samples */
        unsigned int hamdist(uint64_t *x, uint64_t *y)
        {
		    unsigned int dist = 0;

		    for (int i = 0; i < _width; i++)
                dist += _mm_popcnt_u64(x[i] ^ y[i]);

            return dist;
	    }

    private:
        uint64_t *_centroids = NULL;
        uint64_t *_newcentroids = NULL;
        uint64_t *_samples = NULL;
        int       _maxIters;
        uint64_t  _costs; 
        int       _k;
        int       _n;
        int       _width; /* how many uint64_t one binary data occupies */
        int       _bit_width; /* how many bits one binary data occupies */
        float     _threshold; /* in assignment, if second_best_centroid_distance -best_centroid_distance < _threshold,
				 then the point belongs to 2 centroids */

        int *_counts;   /* the number of samples belong to the cluster */
        int *_bitsums;  /* bit counts for all samples belong to the cluster */

        int verbose; /* verbose flag */
        int epsilon; /* if the distance of new center and old center is less than epsilon,
                            treat them as not changed and converged */
        std::vector< std::vector<unsigned int> > _assignment; /* assignment for each point */
        bool assignFlag; /* whether assignment already calculated */

	    /* initialize random centroids */ 
        int initRandCentroids();

};

#endif
