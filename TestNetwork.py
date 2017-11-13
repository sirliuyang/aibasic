import mnist_loader
import avamargo
import myutil
import data_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)


#raw_data = data_loader.getFileMatrix("Train.dat",27)
#raw_training_data = raw_data[0:8000]
#raw_test_data = raw_data[6000:]

#test_data = myutil.toTestData(raw_test_data)
#training_data = myutil.toTrainingData(raw_training_data)

net = avamargo.AvamarGo([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)