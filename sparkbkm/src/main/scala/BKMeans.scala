/*
 * BKMeans for compact binary code
 *
 */

package org.apache.spark.mllib.clustering

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.SparkContext
import org.apache.spark.Logging
import org.apache.spark.annotation.{Experimental, Since}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils

import java.io._

/**
 * K-means clustering with support for multiple parallel runs and random initialization.
 * When multiple concurrent runs are requested, they are executed together with joint passes over
 * the data for efficiency.
 *
 * This is an iterative algorithm that will make multiple passes over the data, so any RDDs given
 * to it should be cached by the user.
 */
class BKMeans private (
    private var bitWidth: Int,
    private var k: Int,
    private var maxIterations: Int,
    private var runs: Int,
    private var assignThreshold: Float) extends Serializable with Logging {

    @native def nativeHamdist(x: Long, y: Long) : Int
    @native def nativeSetCenter(center: Array[Long], sums: Array[Int], count: Int) : Int
    @native def nativeCalcSums(sums: Array[Int], point : Array[Long]) : Int
    @native def nativeArrayHamdist(x: Array[Long], y: Array[Long]) : Int

  /**
   * Constructs a KMeans instance with default parameters: {k: 20, maxIterations: 50, runs: 1,
   * epsilon: 1, seed: random}.
   */
  def this() = this(1024, 20, 50, 1, 1.0F)

  /**
   * bit width of compact binary code.
   */
  def getBitWidth: Int = bitWidth

  /**
   * Set the bit width. Default: 20.
   */
  def setBitWidth(bitWidth: Int): this.type = {
    this.bitWidth = bitWidth
    this
  }

  /**
   * Number of clusters to create (k).
   */
  def getK: Int = k

  /**
   * Set the number of clusters to create (k). Default: 20.
   */
  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  /**
   * Maximum number of iterations to run.
   */
  def getMaxIterations: Int = maxIterations

  /**
   * Set maximum number of iterations to run. Default: 50.
   */
  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  /**
   * :: Experimental ::
   * Number of runs of the algorithm to execute in parallel.
   */
  def getRuns: Int = runs

  /**
   * :: Experimental ::
   * Set the number of runs of the algorithm to execute in parallel. We initialize the algorithm
   * this many times with random starting conditions (configured by the initialization mode), then
   * return the best clustering found over any run. Default: 1.
   */
  def setRuns(runs: Int): this.type = {
    if (runs <= 0) {
      throw new IllegalArgumentException("Number of runs must be positive")
    }
    this.runs = runs
    this
  }

  /**
   * The assignment threshold.
   */
  def getAssignThreshold: Float = assignThreshold

  /**
   * Set the assignment threshold. Default 1024.
   */
  def setAssignThreshold(assignThreshold: Float): this.type = {
    if (assignThreshold <= 0) {
      throw new IllegalArgumentException("Threshold: must be positive")
    }
    this.assignThreshold = assignThreshold
    this
  }

  /**
   * Train a K-means model on the given set of points; `data` should be cached for high
   * performance, because this is an iterative algorithm.
   */
  def run(data: RDD[Array[Long]]): Array[Array[Long]] = {

    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data is not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    val model = runAlgorithm(data)

    // Warn at the end of the run as well, for increased visibility.
    if (data.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }
    model
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(data: RDD[Array[Long]]): Array[Array[Long]] = {

    val sc = data.sparkContext

    val initStartTime = System.nanoTime()

    val numRuns = runs
    val dims = bitWidth/64

    val centers = initRandom(data)
//println("initial centers: " + centers(0))
    val initTimeInSeconds = (System.nanoTime() - initStartTime) / 1e9
    logInfo(s"Initialization with random took " + "%.3f".format(initTimeInSeconds) +
      " seconds.")

    val active = Array.fill(numRuns)(true)
    val costs = Array.fill(numRuns)(0.0)

    var activeRuns = new ArrayBuffer[Int] ++ (0 until numRuns)
    var iteration = 0

    val iterationStartTime = System.nanoTime()

    // Execute iterations of Lloyd's algorithm until all runs have converged
    while (iteration < maxIterations && !activeRuns.isEmpty) {
      type WeightedPoint = (Array[Int], Int)
      def mergeContribs(x: WeightedPoint, y: WeightedPoint): WeightedPoint = {
        ((x._1, y._1).zipped.map(_ + _), x._2 + y._2)
      }

      println(" Iteration " + iteration)
      val activeCenters = activeRuns.map(r => centers(r)).toArray
      val costAccums = activeRuns.map(_ => sc.accumulator(0.0))

      val bcActiveCenters = sc.broadcast(activeCenters)

      // Find the sum and count of points mapping to each center
      val totalContribs = data.mapPartitions { points =>
        val thisActiveCenters = bcActiveCenters.value
        val runs = thisActiveCenters.length
        val k = thisActiveCenters(0).length

        val sums = Array.fill(runs, k)(Array.fill(bitWidth){0})
        val counts = Array.fill(runs, k)(0)

        points.foreach { point =>
          (0 until runs).foreach { i =>
            val (bestCenter, cost) = findClosest(thisActiveCenters(i), point)
//println("bestCenter: " + bestCenter + " [" + thisActiveCenters(i)(bestCenter).mkString(", ") + "] point: "+ point+ " cost: " + cost);
            costAccums(i) += cost*cost
            nativeCalcSums(sums(i)(bestCenter), point)
            /* caculate sums if not use native function */
            /*sums(i)(bestCenter) = sums(i)(bestCenter).zipWithIndex.map { 
                case (e,i) => if ((((point(i/64) >>> (63-i))) & 1) != 0) e+1 else e}*/
            counts(i)(bestCenter) += 1
//println("sums: " + sums(i)(bestCenter) + " counts[" + bestCenter + "]:" + counts(i)(bestCenter));
          }
        }

        val contribs = for (i <- 0 until runs; j <- 0 until k) yield {
          ((i, j), (sums(i)(j), counts(i)(j)))
        }
        contribs.iterator
      }.reduceByKey(mergeContribs).collectAsMap()

      //println(" find center ")
      bcActiveCenters.unpersist(blocking = false)

      // Update the cluster centers and costs for each active run
      for ((run, i) <- activeRuns.zipWithIndex) {
        var changed = false
        var j = 0
        while (j < k) {
          val (sum, count) = totalContribs((i, j))
//println("sum: " + sum.mkString(", ") + " count: " + count)
          if (count == 0) {
            centers(run)(j).map(s => 0);
          } else {
            var favorzero = true; /* if two point is 0110 and 1011, how to compute the centorid?
                                  They are multiple choices to make the hamming distance same,
                                  0110 or 1011 both are ok, we prefer 0110 */

            var newCenter = Array.fill(dims)(0L)
            nativeSetCenter(newCenter, sum, count)
            /* update center if not use native function */
            /*sum.zipWithIndex.foreach { case(x,i) => 
              var bitsum2 = x * 2
              if (bitsum2 > count) newCenter(i/64) |= 1L << (63-(i%64))
              else if (bitsum2 == count) {
                if (favorzero) {
                  favorzero = false;
                } else {
                  newCenter(i/64) |= 1L << (63-(i%64))
                  favorzero = true;
                }
              }
            }*/
            if (newCenter.deep != centers(run)(j).deep) {
              changed = true
            }
            centers(run)(j) = newCenter
          }
          j += 1
        }
//println("-------- new centers: [" + centers(run).mkString(", ") + "]")
        if (!changed) {
          active(run) = false
          logInfo("Run " + run + " finished in " + (iteration + 1) + " iterations")
        }
        costs(run) = math.sqrt(costAccums(i).value)
      }


      //println(" update center ")
      activeRuns = activeRuns.filter(active(_))
      iteration += 1
    }

    val iterationTimeInSeconds = (System.nanoTime() - iterationStartTime) / 1e9
    logInfo(s"Iterations took " + "%.3f".format(iterationTimeInSeconds) + " seconds.")

    if (iteration == maxIterations) {
      println("BKMeans reached the max number of iterations: " + maxIterations)
    } else {
      println("BKMeans converged in " + iteration + " iterations.")
    }

    val (minCost, bestRun) = costs.zipWithIndex.min

    logInfo(s"The cost for the best run is $minCost.")

    centers(bestRun)    
  }

  /**
   * Initialize `runs` sets of cluster centers at random.
   */
  private def initRandom(data: RDD[Array[Long]])
  : Array[Array[Array[Long]]] = {
    // Sample all the cluster centers in one pass to avoid repeated scans
    val sample = data.takeSample(true, runs * k).toSeq
    Array.tabulate(runs)(r => sample.slice(r * k, (r + 1) * k).toArray)
  }

  /**
   * Returns the index of the closest center to the given point, as well as the distance.
   */
  private[mllib] def findClosest(
      centers: TraversableOnce[Array[Long]],
      point: Array[Long]): (Int, Double) = {
    var bestDistance = Int.MaxValue
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val distance: Int = nativeArrayHamdist(center, point)
      //val distance: Int = HammingDistance(center, point)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    //println("-------- bestcenters: [" + bestIndex + "] , distance = " + bestDistance)
    (bestIndex, bestDistance)
  }

  /**
   * Returns all the index of the centers which distance <= best_distance/assignThreshold.
   */
  private[mllib] def findCloseCenters(
      centers: TraversableOnce[Array[Long]],
      point: Array[Long]): Array[Int] = {
    var bestDistance = Int.MaxValue
    var bestIndex = 0
    var buf = scala.collection.mutable.ListBuffer.empty[(Int, Int)]
    var i: Int = 0
    centers.foreach { center =>
      val distance: Int = nativeArrayHamdist(center, point)
      val di = (distance, i)
      buf += di
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    //println("-------- bestcenters: [" + bestIndex + "] , distance = " + bestDistance)
    buf.filter{ x => var (u, _) = x; u*assignThreshold <= bestDistance }.sorted.map { x => var (_, v) = x; v}.toArray
  }
}


/**
 * Top-level methods for calling K-means clustering.
 */
object BKMeans {

  System.loadLibrary("nativeBKMeans")

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data training points stored as `RDD[Array]`
   * @param bitWidth bit width of the binary data, default 1024
   * @param k number of clusters, default 20
   * @param maxIterations max number of iterations, default 50
   * @param runs number of parallel runs, defaults to 1. The best model is returned.
   * @param assignThreshold threshold of assignment, default 1.0 distance <= best_distance/assignThreshold
   */
  def train(
      data: RDD[Array[Long]],
      bitWidth: Int,
      k: Int,
      maxIterations: Int,
      runs: Int,
      assignThreshold: Float): Array[Array[Long]] = {
    new BKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .setBitWidth(bitWidth)
      .setAssignThreshold(assignThreshold)
      .run(data)
  }

  /**
   * Trains a k-means model using the given set of parameters.
   *
   * @param data training points stored as `RDD[Array]`
   * @param k number of clusters
   * @param maxIterations max number of iterations
   * @param runs number of parallel runs, defaults to 1. The best model is returned.
   */
  def train(
      data: RDD[Array[Long]],
      k: Int,
      maxIterations: Int,
      runs: Int): Array[Array[Long]] = {
    new BKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .setRuns(runs)
      .run(data)
  }

  /**
   * Trains a k-means model using specified parameters and the default values for unspecified.
   */
  def train(
      data: RDD[Array[Long]],
      k: Int,
      maxIterations: Int): Array[Array[Long]] = {
    new BKMeans().setK(k)
      .setMaxIterations(maxIterations)
      .run(data)
  }

  /**
   * Returns the cluster index that a given point belongs to.
   */
  def predict(centers: Array[Array[Long]], point: Array[Long]) : Int = {
    new BKMeans().findClosest(centers, point)._1
  }

  /**
   * Returns the K-means cost of a given point against the given cluster centers.
   */
  def pointCost(centers: Array[Array[Long]], point: Array[Long]) : Double = {
    new BKMeans().findClosest(centers, point)._2
  }

    /**
     * Return the K-means cost (sum of squared distances of points to their nearest center) for this
     * model on the given data.
     */
  def computeCost(data: RDD[Array[Long]], centers: Array[Array[Long]]): Double = {
    data.map(p => { val cost = pointCost(centers, p); cost*cost }).sum
  }

  /**
   * Returns all the cluster index that a given point belongs to.
   */
  def calcAssignment(centers: Array[Array[Long]], point: Array[Long], assignThreshold: Float) : Array[Int] = {
    new BKMeans().setAssignThreshold(assignThreshold).findCloseCenters(centers, point)
  }

  /**
   * Save centers to file.
   */
  def saveCentersLocal(centers: Array[Array[Long]], file: String) : Int = {
    val pw = new PrintWriter(new File(file))
    centers.foreach {center => pw.write(center.mkString(" ") + "\n")}
    pw.close()
    0
  }

  /**
   * Save centers to Spark.
   */
  def saveCentersSpark(sc: SparkContext, centers: Array[Array[Long]], path: String) : Int = {
    val lines= centers.map {center => center.mkString(" ") }
    sc.makeRDD(lines).coalesce(1,true).saveAsTextFile(path)
    0
  }

  /**
   * Load centers from file.
   */
  def loadCenters(sc: SparkContext, file: String) : Array[Array[Long]] = {
    val data = sc.textFile(file)
    var parsedCenters = data.map(s=> s.split(" ").map(s=> { var x = BigInt(s); x.longValue()} ) )
    parsedCenters.collect
  }
  /**
   * Returns the Hamming distance between two compact binary code
   */
  def HammingDistance(
      v1: Array[Long],
      v2: Array[Long]): Int = {
    v1.zip(v2).map(s => java.lang.Long.bitCount(s._1 ^ s._2)).sum
  }

}

