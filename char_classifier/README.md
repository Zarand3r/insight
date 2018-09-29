Instructions
============
retrain.py retrains the inception network on data supplied from TrainingData
label_image.py labels the inception model, but should work with any model

Prepare models by adding features in models.py and run the script to generte the models.json file

Under CNN, you will find the neural networks. 
mnist_cnn.py trains on the mnist dataset in TrainingData and outputs a model into models
emnist_cnn.py trains on the emnist dataset in TrainingData and outputs a model into models
freeze.py generates the frozen graph 
predictor.py feeds new input into the models (either frozen graph or checkpoint) and produces an output
detect_letters runs computer vision to detect letters in an image of hand-drawn text and classifies the letters. 
	goals is to adapt this to numbers too