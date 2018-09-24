import argparse 
import tensorflow as tf


def read_tensor_from_image_file(file_name,
                                input_height=28,
                                input_width=28,
                                input_mean=0,
                                input_std=255,
                                channels=3):
  input_name = "file_reader"
  output_name = "normalized"

  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=channels, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels=channels, name="jpeg_reader")

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        # tf.import_graph_def(graph_def, name="prefix")
        tf.import_graph_def(graph_def)
    return graph

if __name__ == '__main__':
    model_path = '../models/mnist_model/frozen_model.pb'
    input_node = 'Reshape'
    output_node = 'softmax_tensor'
    test = "../imagedata/test.png"


    input_name = "import/" + input_node
    output_name = "import/" + output_node


    graph = load_graph(model_path)

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name)
        
    # We access the input and output nodes 
    # x = graph.get_tensor_by_name(input_node)
    # y = graph.get_tensor_by_name(output_node)
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    x = input_operation.outputs[0]
    y = output_operation.outputs[0]

    # input parameters are to reshape the image array into a tensor with the right dimensions to feed into neural network
    image = read_tensor_from_image_file(test, input_height = 28, input_width = 28, channels = 1)
    # print(image)
        
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # Note: we don't nee to initialize/restore anything
        # There is no Variables in this graph, only hardcoded constants 
        y_out = sess.run(y, feed_dict={
            x: image
        })
        # I taught a neural net to recognise when a sum of numbers is bigger than 45
        # it should return False in this case
        print(y_out) # [[ False ]] Yay, it works!