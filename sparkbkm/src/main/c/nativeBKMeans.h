/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_apache_spark_mllib_clustering_BKMeans */

#ifndef _Included_org_apache_spark_mllib_clustering_BKMeans
#define _Included_org_apache_spark_mllib_clustering_BKMeans
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeHamdist
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeHamdist
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeArrayHamdist
 * Signature: ([J[J)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeArrayHamdist
  (JNIEnv *, jobject, jlongArray, jlongArray);

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeSetCenter
 * Signature: ([J[II)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeSetCenter
  (JNIEnv *, jobject, jlongArray, jintArray, jint);

/*
 * Class:     org_apache_spark_mllib_clustering_BKMeans
 * Method:    nativeCalcSums
 * Signature: ([I[j)I
 */
JNIEXPORT jint JNICALL Java_org_apache_spark_mllib_clustering_BKMeans_nativeCalcSums
  (JNIEnv *, jobject, jintArray, jlongArray);

#ifdef __cplusplus
}
#endif
#endif
