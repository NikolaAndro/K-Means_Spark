import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

// Load and parse the data
val data = sc.textFile("amazon1.txt")
val parsedData = data.map(s => Vectors.dense(s.split('\t').map(_.toDouble))).cache()

// Cluster the data into two classes using KMeans
val numClusters = 2
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)

// Evaluate clustering by computing Within Set Sum of Squared Errors
val WSSSE = clusters.computeCost(parsedData)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Save and load model
clusters.save(sc, "KMeansModel")
val sameModel = KMeansModel.load(sc, "KMeansModel")

// Shows the result.
println("Cluster Centers: ")
sameModel.clusterCenters.foreach(println)
