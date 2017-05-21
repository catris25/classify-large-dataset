import time
start_time = time.time()

# SAMPLING TRAINING SET
# print("# SAMPLING TRAINING SET")
# input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
# import sampling_trainingset as train
# training_sampled_dir = train.sampling(input_file)
# print("training sampled:", training_sampled_dir)

# # CLUSTERING TRAINING SET
# print("# CLUSTERING TRAINING SET")
# import clustering_k_means as clsr
# # input_file = training_sampled_dir
# input_file = "output/training_sampled/2017-05-07 23:12:26/training_set.csv"
# k_size = 3
# training_clustered_dir = clsr.kmeans(input_file, k_size)
# print("training clustered:",training_clustered_dir)

# CLUSTERING TRAINING SET
print("# CLUSTERING TRAINING SET")
import clustering_k_means_sklearn as clsr
# input_file = training_sampled_dir
input_file = "output/training_sampled/2017-05-07 23:12:26/training_set.csv"
k_size = 25
training_clustered_dir = clsr.clustering(input_file, k_size)
print("training clustered:",training_clustered_dir)


# SAMPLING TESTING SET
# print("# SAMPLING TESTING SET")
# input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
# import sampling_testingset as test
# n_size = 2000
# testing_sampled_dir = test.sampling(input_file, n_size)
# print("testing sampled:", testing_sampled_dir)

# CLUSTERING TESTING SET
print("# CLUSTERING TESTING SET")
# testing_file = testing_sampled_dir
testing_file = "output/testing_sampled/2017-05-07 23:12:32/testing-set.csv"
training_clustered_file = training_clustered_dir
import clustering_testingset as clsrtest
testing_clustered_dir = clsrtest.cluster_test(training_clustered_file, testing_file)
print("testing clustered:", testing_clustered_dir)

# # CLASSIFICATION NAIVE BAYES
# print("# CLASSIFICATION NAIVE BAYES")
# import classify_nb as nb
# training_dir = training_clustered_dir
# testing_dir = testing_clustered_dir
# tp_nb, all_nb = nb.classify_all(training_dir,testing_dir, k_size)
#
# # CLASSIFICATION K NEAREST NEIGHBOR
# print("# CLASSIFICATION K NEAREST NEIGHBOR")
# import classify_knn as knn
# training_dir = training_clustered_dir
# testing_dir = testing_clustered_dir
# n_size = 7
# tp_knn, all_knn = knn.classify_all(training_dir,testing_dir, k_size, n_size)

# # CLASSIFICATION LOGISTIC regression
print("# CLASSIFICATION LOGISTIC regression")
import classify_logistic_regression as lr
training_dir = training_clustered_dir
testing_dir = testing_clustered_dir
tp_lr, all_lr = lr.classify_all(training_dir,testing_dir, k_size)


# RESULT
# print("-------------------")
# print("# RESULT")
# print("cluster size =",k_size)
# print("Naive Bayes")
# print(tp_nb,"/",all_nb)
# print("KNN")
# print(tp_knn,"/",all_knn)
print("LR")
print(tp_lr,"/",all_lr)

time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
