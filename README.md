Background
==========

`K-Means` is an unsupervised learning technique that groups n data into k meaningful subclasses (clusters), minimizes the intra-differences and maximizes the inter-differences. That is if two data points are in the same cluster, their difference should be small. K-Means is among the most popular and simplest clustering methods.

This is the detailed explanation of K-Means algorithm. The first step is choose K data points randomly ad initial centroids. Then repeat the training. The training has two steps. Step one is assign step: for each data point x in the dataset, compute the distance between x and each centroid, find the shortest distance the assign x to the nearest centroid. After all data points assigned, each centroid has its own data points, step 2 - mean step is applied. Suppose a specific centroid has 1000 data points, the compute the mean of these 1000 data points, and set the mean as the new centroid. Repeat the training process until the centroids not change anymore, which is the assignment of all data points are fixed.

Find the nearest centroid need compute the distances to all centroids. In mathematics, the Euclidean distance is the straight-line distance between two points in Euclidean space. It’s natural to use Euclidean distance, although other distances such as cosine distance, Manhattan distance, Minkowski distance, Euclidean distance is most often used. The mean computation is also straight forward. Use the arithmetic mean – compute the mean on each dimension.

Image duplication dectection as K-Means usage example
=====================================================

An example usage of K-Means is image duplicate detection. Sometimes, in e-commerce marketplace small seller may copy pictures from other big sellers and add their own logo, use them as product pictures. It’s harmful to the e-commerce platform and big sellers, K-Means can be used to detect duplicated images. The traditional way to use K-Means for image clustering has two steps. The first step is feature extraction, using convolutional neural network to get the feature vector from the image. We already talked about CNN and different models. Often the dimension of the feature vector is high, more than four thousand. Each dimension is float or double.The second step is clustering, use K-Means with the feature vector. K-Means algorithm has been implemented in many frameworks, such as Spark MLlib.

There are some challenges to handle massive images. The first challenge is huge memory usage. One feature vector will use 16 kilobytes. 4096 multiply by the size of float, we get 16K. One million images will use 16 gigabytes, one billion images will use 16 terabytes. 1.2m images is not enough for real business, they always have hundreds million images which can’t fit into a single machine.

The second challenge is compute intensive. The standard K-Means, the most computation is finding the nearest centroid for each point in the dataset. This step need to compute the distance between every point in all the centroids and pick the closest centroid. When both n and k are large, the computation is extremely large. And the distance computation needs square and square root operation.

We ca improve it by using binary K-Means. The images are turned to binary code. Each image can be represented by a short binary code, 1024 or lower bits. The binary code is similarity–preserving hash. If two images are similar, their binary code should also be similar. There has been considerable amount of work on learning similarity preserving binary hash codes for Internet images.
Distance computation between these binary codes can be done using fast hamming distance. The hamming distance is the number of different bits. For example these two binary codes have 2 different bits, the hamming distance between them is 2. Hamming distance can be commutated by exclusive OR then count the number of bit 1.

Building BKMeans
================

```
cd src
make
```

Using BKMeans
=============

```
usage: ./bkmeans.bin [B] [-b bit_width] [-k cluster_number] [-i iteration] [-f datafile] [-v] [-h]
	-B datafile byte format like '184 81 173 232' instead of defalut uint64_t format '2867679328082739030'
	-b bit_width, default 1024
	-k the number of clusters (centroids), default 20
	-i iterations, default 50
	-f datafile name
	-v verbose
	-c computeCost
	-h help

./bkmeans.bin -k 10 -f ../data/imagenet_codes_1 
n=1300
------------------ begin --------------
----------- Iteration: 1, WSSSE: 384822.000000 10976.568225 ----------
----------- Iteration: 2, WSSSE: 377935.000000 10785.355024 ----------
......................................................................
----------- Iteration: 41, WSSSE: 371802.000000 10616.459014 ----------
----------- Iteration: 42, WSSSE: 371802.000000 10616.459014 ----------

INFO: algorithm fully converged in 42 iterations
--------- Train finished in 1 seconds -----------
```


Using BKMeans in Spark MLlib
============================
BKMeams can also be used with Spark MLLib, the stepa are as folows:

* generate bkmeans package 
 ```
 $ sbt package
 ```
 bkmeans-project_2.10-1.0.jar will be genarated

* compile native C code
 ```$ cd sparkbkm/src/main/c
 $ gcc -shared -fpic -o libnativeBKMeans.so -msse4.2 -I/usr/local/jdk1.7.0_79/include -I/usr/local/jdk1.
7.0_79/include/linux nativeBKMeans.c
```

* compile test code
 ```$ cd sparkbkm/src/test/scala
 $ mkdir lib; cd lib; ln -s sparkbkm/target/scala-2.10/bkmeans-project_2.10-1.0.jar .
 $ sbt package
 ```

* run
``` $ spark-submit --files /home/jimincai/dev/Bkmeans/sparkbkm/src/main/c/libnativeBKMeans.so --jars ~/dev/
Bkmeans/sparkbkm/target/scala-2.10/bkmeans-project_2.10-1.0.jar --master local[*] target/scala-2.10/bkme
anstest_2.10-1.0.jar -k 20 -i 50 -f file:///Bkmeans/data/data1
```
