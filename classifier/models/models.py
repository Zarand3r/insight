import json

data = {}  
data["inception_v3_2016_08_28"] = {
    "input_height": 299,
    "input_width": 299,
    "model_file": "inception_v3_2016_08_28/inception_v3_2016_08_28_frozen.pb",
    "script": "",
    "checkpoint_file": "",
    "label_file": "inception_v3_2016_08_28/labels.txt",
    "input_node": "input",
    "output_node": "InceptionV3/Predictions/Reshape_1"
}  

data["mnist_model"] = {
    "input_height": 28,
    "input_width": 28,
    "script": "mnist_cnn",
    "model_file": "mnist_model/frozen_model.pb",
    "checkpoint_file": "mnist_model/model.ckpt-20000.meta",
    "label_file": "mnist_model/labels.txt",
    "input_node": "Reshape",
    "output_node": "softmax_tensor"
}  

data["emnist_model"] = {
    "input_height": 28,
    "input_width": 28,
    "script": "emnist_cnn",
    "model_file": "emnist_model/frozen_model.pb",
    "checkpoint_file": "emnist_model/model.ckpt-20000.meta",
    "label_file": "emnist_model/labels.txt",
    "input_node": "Reshape",
    "output_node": "softmax_tensor"
}  

data["emnist_cnn_model"] = {
    "input_height": 28,
    "input_width": 28,
    "script": "emnist_classifier",
    "model_file": "emnist_cnn_model/frozen_model.pb",
    "checkpoint_file": "emnist_cnn_model/model.ckpt-12000.meta",
    "label_file": "emnist_cnn_model/labels.txt",
    "input_node": "Reshape",
    "output_node": "softmax_tensor"
}  



# mobilenet_v1_1.0_224

with open("models.json", "w") as outfile:  
    json.dump(data, outfile)