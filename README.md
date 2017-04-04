# classify-large-dataset
My attempts at doing classification on large datasets using some data mining algorithms.

# how to use this program
I know it's kind of weird, but whatever. We'll think about it later again.
1. First, do the stratified random sampling and decide how many data you want to sample, by running stratified-sampling.py. There, you'll obtain the files in results/sampled/*timestamp*. There are going 3 files here, all in .csv format. They are the dataset itself, the summary (stats) of the data, and the number of data taken each class.
2. Then, use the obtained dataset from the first step to create clusters. Decide how many k you want to have and run k-means-clustering.py. Here, you'll get k number of files, just as many as your clusters. Each cluster data is stored in each own file. The data are stored in results/clustered/*timestamp*.
3. Third, after you get your data sampled and clustered from the two steps above, you want to classify them and see how good they are. But first, you want to obtain the testing test first. So, you run testing-sampling.py, to get the data used for testing.  
