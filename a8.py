from neural import *
import time

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

xorn = NeuralNet(2, 1, 1)
sTime = time.process_time()
xorn.train(data=xor_training_data, iters=3000) #Point of convergence constantly changes
print(xorn.test_with_expected(xor_training_data))
print(f"Elapsed Time: {time.process_time() - sTime}")

print("<<<<<<<<<<<<<< XOR 2 >>>>>>>>>>>>>>\n")

bigXorn = NeuralNet(2, 8, 1)
sTime = time.process_time()
bigXorn.train(data=xor_training_data, iters=2000) #Far lower error
print(bigXorn.test_with_expected(xor_training_data))
print(f"Elapsed Time: {time.process_time() - sTime}") #Less iterations longer clock time

print("<<<<<<<<<<<<<< XOR 3 >>>>>>>>>>>>>>\n")

smallXorn = NeuralNet(2, 1, 1)
sTime = time.process_time()
smallXorn.train(data=xor_training_data, iters=1000) #High error, barely decreases
print(smallXorn.test_with_expected(xor_training_data))
print(f"Elapsed Time: {time.process_time() - sTime}")

print("<<<<<<<<<<<<<< POLITICS >>>>>>>>>>>>>>\n")

partyTrainData = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0])
]

partyTestData = [
    [1.0, 1.0, 1.0, 0.1, 0.1],
    [0.5, 0.2, 0.1, 0.7, 0.7],
    [0.8, 0.3, 0.3, 0.3, 0.8],
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6]
]

partyN = NeuralNet(5, 8, 1)
sTime = time.process_time()
partyN.train(data=partyTrainData, iters=1000)
print(partyN.test(partyTestData))
print(f"Elapsed Time: {time.process_time() - sTime}")

