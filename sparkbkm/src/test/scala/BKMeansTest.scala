import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.BKMeans
import org.apache.spark.mllib.linalg.Vectors
object BKMeansTest {
    def main (args: Array[String]) {
         val usage = """
                  Usage: BKMeansTest [-b bit_width] [-k cluster_number] [-i iteration] [-f datafile] 

                     -b bit_width, default 1024
                     -k the number of clusters (centroids), default 20
                     -i iterations, default 50
                     -f datafile name
          """

         if (args.length == 0) println(usage)
         val arglist = args.toList
         type OptionMap = Map[Any, Any]


         def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
           def isSwitch(s : String) = (s(0) == '-')
           list match {
             case Nil => map
             case "-b" :: value :: tail =>
               nextOption(map ++ Map("bitwidth" -> value.toInt), tail)
             case "-k" :: value :: tail =>
               nextOption(map ++ Map("k" -> value.toInt), tail)
             case "-i" :: value :: tail =>
               nextOption(map ++ Map("iteration" -> value.toInt), tail)
             case "-f" :: value :: tail =>
               nextOption(map ++ Map("file" -> value), tail)
             case option :: tail => println("Unknown option "+option) 
                 exit(1) 
           }
         }

         val options = nextOption(Map(),arglist)

         println("----------- begin ---------------");
         val conf = new
             SparkConf().setAppName("Spark MLlib BKMeansTest")
         val sc = new SparkContext(conf)
         
         //var data = sc.textFile("file:///home/jimincai/dev/Bkmeans/data/bin.ft2")
         var data = sc.textFile(options("file").toString)

         var parsedData = data.map(s=> s.split(" ").map(s=> { var x = BigInt(s); x.longValue()} ) ).cache()
         var count = parsedData.count().toInt

         println("----------- parsed data, count = " + count + " ---------------");

         // Cluster the data into two classes using KMeans
         var numClusters =  20
         if (options.contains("k")) numClusters = options("k").toString.toInt
         var numIterations = 50
         if (options.contains("iteration")) numIterations = options("iteration").toString.toInt

         val clusters = BKMeans.train(parsedData, numClusters, numIterations)
         println("----------- end ---------------");

         //parsedData.take(count).foreach( s => { var bestindex = BKMeans.predict(clusters, s); println(bestindex)} )
    }
}
