'''
Trying to make a XOR neural net using neurolab
'''
import neurolab as nl


input = [
    [0, 1],
    [1, 1],
    [1, 0],
    [0, 0]
]
target = [
    [1],
    [0],
    [1],
    [0],
]

# Make 6 inputs, 20 neurons and 6 outputs
net = nl.net.newff(
    [
        [0, 1], [0, 1],
        #[0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
    ],
    [2, 1]
)
err = net.train(input, target, show=15, goal=0.01)

print net.sim([[0, 1]])
print net.sim([[1, 1]])
print net.sim([[1, 0]])
print net.sim([[0, 0]])
