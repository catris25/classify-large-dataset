# Lia Ristiana
# this is the main program? WHAT?
# THIS CODE FEELS SO STUPID
# IF ONLY YOU CAN FIND A BETTER WAY, I'D BE HAPPIER
# BUT IT WORKS FOR NOW, SO MAYBE I'D BETTER SHUT UP TOO

# custom_k_means file --> cluster the data into smaller partition
main_k = 8
main_sampled_dir_name = "2017-05-04 23:23:02"
import custom_k_means
main_clustered_set_dir = custom_k_means.clustered_set_dir

# testing_sampling --> sample the data from the main dataset for testing purposes
# main_n_testing_size = 1000
# import testing_sampling
# main_testing_set_dir = testing_sampling.testing_set_dir
main_testing_set_dir = "results/testing/2017-05-05 01:47:07/testing-set.csv"

# cluster-decision --> decide which cluster the testing set is closest to
import cluster_decision

main_testing_dir = cluster_decision.testing_set_dir
main_training_dir = main_clustered_set_dir

# classify_nb
import classify_nb
