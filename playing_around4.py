'''
Working with my personal data again, using sk-learn
'''

from sklearn import tree, linear_model, neighbors
import pandas as pd
import numpy


data_labels = ["Happiness", "Motivation", "Flexibility", "Strength", "Endurance", "Relationships"]
data_frame = pd.read_csv("personal_stats2.csv")
data_frame = data_frame[data_labels]

# Get 80% of our dataset
index_at_80_percent = int(len(data_frame) * .8)

# Get the first 80% as input and the following day as the target result
training_input = data_frame[:index_at_80_percent]
training_target = data_frame[1:index_at_80_percent + 1]

# The final 20% same as above, current day as input next day as expected output
test_input = data_frame[index_at_80_percent + 1: len(data_frame) - 1]
test_target = data_frame[index_at_80_percent + 2:]

# Uncomment to select a method
#clf = tree.DecisionTreeClassifier()
clf = linear_model.LinearRegression(copy_X=True, normalize=False, fit_intercept=True)                                 # 956
#clf = linear_model.RANSACRegressor(linear_model.LinearRegression())   # 2000
#clf = linear_model.Lasso()                                            # 976
#clf = neighbors.KNeighborsRegressor()                                 # 998
#clf = linear_model.LassoLars(alpha=.01)                               # 944
#clf = linear_model.OrthogonalMatchingPursuit()                        # 941
#clf = linear_model.ElasticNet(alpha=.5, l1_ratio=0.5, tol=0.01)       # 955
#clf = linear_model.PassiveAggressiveRegressor()

# Implement the method and print the results, very simple, only takes into account previous days record
clf = clf.fit(training_input, training_target)

print ""
print "==========================================================================="
print "Given the previous day's result, what can we predict tomorrows stats to be?"
print "==========================================================================="
print "\nPredict stats for the day after [10, 10, 10, 10, 10, 10]"
prediction = clf.predict([[10, 10, 10, 10, 10, 10]])
print pd.DataFrame(prediction, columns=data_labels)

print "\nPredict stats for the day after [1, 1, 1, 1, 1, 1]"
prediction = clf.predict([[1, 1, 1, 1, 1, 1]])
print pd.DataFrame(prediction, columns=data_labels)


#=============================================================================
# Compare predictions to actual stats
#
# Uncomment to view!
#=============================================================================
# total_difference_less_than_5 = 0
# total_difference_greater_than_15 = 0
# total_difference = 0
#
# for index, row in enumerate(test_input.values):
#     test_result = numpy.array([int(n) for n in clf.predict(row)[0]])
#     difference = sum(abs(x - y) for x, y in zip(test_result, test_target.values[index]))
#     total_difference += difference
#
#     print test_target.values[index]
#     print test_result
#     print "Difference:", difference
#     print ""
#
#     #print total_difference, difference
#
#     if difference < 5:
#         total_difference_less_than_5 += 1
#     if difference > 15:
#         total_difference_greater_than_15 += 1
#
# print ""
# print "Total differences < 5 -->", total_difference_less_than_5
# print "Total differences > 15 -->", total_difference_greater_than_15
# print "Total differences ==", total_difference
