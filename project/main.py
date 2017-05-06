import time
start_time = time.time()

input_file="/home/lia/Documents/FINAL-GEMASTIK/training.csv"
import sampling.trainingset_sampling as train
sampled_training_dir = train.sampling(input_file)



time_elapsed = time.time() - start_time
print("--- %s seconds ---" % (time_elapsed))
