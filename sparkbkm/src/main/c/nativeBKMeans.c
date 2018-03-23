#include <ctype.h>
#include <string.h>
#include <stdint.h>
#include <jni.h>
#include <nmmintrin.h>
#include "nativeBKMeans.h"

int hamdist(uint64_t *x, uint64_t *y, int width)
{
    int dist = 0, i;
    
    for (i = 0; i < width; i++)
        dist += _mm_popcnt_u64(x[i] ^ y[i]);

    return dist;
}

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeHamdist
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeHamdist
  (JNIEnv *env, jobject obj, jlong x, jlong y) {
    return _mm_popcnt_u64(x ^ y);
}

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeArrayHamdist
 * Signature: ([J[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeArrayHamdist
  (JNIEnv *env, jobject obj, jlongArray x, jlongArray y) {
    int sum = 0, i;
    jsize len = (*env)->GetArrayLength(env, x);
    jlong *xarr = (*env)->GetLongArrayElements(env, x, 0);
    jlong *yarr = (*env)->GetLongArrayElements(env, y, 0);

    for (i=0; i<len; i++) {
        sum += _mm_popcnt_u64(xarr[i] ^ yarr[i]);
    }

    (*env)->ReleaseLongArrayElements(env, x, xarr, 0);
    (*env)->ReleaseLongArrayElements(env, y, yarr, 0);

    return sum;
}

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeSetCenter
 * Signature: ([J[II)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeSetCenter
  (JNIEnv *env, jobject obj, jlongArray center, jintArray sums, jint count) {
    jlong *c = (*env)->GetLongArrayElements(env, center, 0);
    jint *s = (*env)->GetIntArrayElements(env, sums, 0);
    jsize len = (*env)->GetArrayLength(env, sums);
    int _width = len/64;
    
    int bitsum2, w, j;

    for (w=0; w<_width; w++) {
        for (j=0; j<64; j++) {
            bitsum2 = s[w*64 +j] * 2;
            if (bitsum2 >= count) {
                c[w] |= 1L<<j;
            }
        }
    }

    (*env)->ReleaseLongArrayElements(env, center, c, 0);
    (*env)->ReleaseIntArrayElements(env, sums, s, 0);
}

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeCalcSums
 * Signature: ([I[j)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeCalcSums
  (JNIEnv *env, jobject obj, jintArray sums, jlongArray point) {
    jlong *p = (*env)->GetLongArrayElements(env, point, 0);
    jint *s = (*env)->GetIntArrayElements(env, sums, 0);
    jsize len = (*env)->GetArrayLength(env, sums);
    int _width = len/64, w,j;

    for (w=0; w<_width; w++) {
        for (j=0; j<64; j++) {
            if (p[w] & (1L<<j)) {
                s[w*64 +j]++;
            }
        }
    }
    
    (*env)->ReleaseLongArrayElements(env, point, p, 0);
    (*env)->ReleaseIntArrayElements(env, sums, s, 0);

    return 1;
}

