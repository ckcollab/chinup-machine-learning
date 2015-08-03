'''
Playing with my personal stats trying to draw some conclusion of some kind
'''

import neurolab as nl
import pandas as pd


data_frame = pd.read_csv("personal_stats2.csv")
data_frame = data_frame[["Happiness", "Motivation", "Flexibility", "Strength", "Endurance", "Relationships"]]

# Get 80% of our dataset
index_at_80_percent = int(len(data_frame) * .8)

# Get the first 80% as input and the following day as the target result
training_input = data_frame[:index_at_80_percent]
training_target = data_frame[1:index_at_80_percent + 1]

# The final 20% same as above, current day as input next day as expected output
test_input = data_frame[index_at_80_percent + 1: len(data_frame) - 1]
test_target = data_frame[index_at_80_percent + 2:]

training_input = training_input / 10
training_target = training_target / 10
test_input = test_input / 10
test_target = test_target / 10

# Make 6 inputs, 20 neurons and 6 outputs
net = nl.net.newff(
    [
        [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]
        #[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
    ],
    [20, 6]
)
err = net.train(training_input, training_target, rr=0.05, show=15, epochs=1000)

print "done"
