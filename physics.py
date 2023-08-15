

from generate_graph import GenerateGraphTuple, GenerateGraphVisual

from graph_nets import utils_np
from graph_nets import blocks
from graph_nets import utils_tf
from graph_nets.demos import models
from matplotlib import pyplot as plt
import numpy as np


import time
import tensorflow as tf
import sonnet as snt
import open3d as o3d



# Create the model.
model = models.EncodeProcessDecode(node_output_size=3)

# Optimizer.
learning_rate = 1e-3
optimizer = snt.optimizers.Adam(learning_rate)
# step_op = optimizer.minimize(loss_op_tr)

def get_data(graph_tuple, batch_num):
  input_gaph_tr, target_graph_tr, base_graph_t = graph_tuple.get_graph_batch(batch_num)
  return input_gaph_tr, target_graph_tr, base_graph_t



def create_loss(target_nodes, outputs):
  """Create supervised loss operations from targets and outputs.

  Args:
    target_op: The target velocity tf.Tensor.
    output_ops: The list of output graphs from the model.

  Returns:
    A list of loss values (tf.Tensor), one per output op.
  """

  # if not isinstance(outputs, collections.Sequence):
  #   outputs = [outputs]

  losss = [
      tf.math.reduce_mean(
          tf.math.reduce_sum((output.nodes - target_nodes[..., 3:6])**2+ (output.nodes - target_nodes[..., 0:3])**2, axis=-1))
      for output in outputs
  ]
  return tf.stack(losss)

def update_step(input_graph_tr, target_tr, num_processing_steps_tr=1):
  with tf.GradientTape() as tape:
    outputs_tr = model(input_graph_tr, num_processing_steps_tr)
    loss_tr = create_loss(target_tr, outputs_tr)
    loss_tr = tf.math.reduce_sum(loss_tr) / num_processing_steps_tr
  gradients = tape.gradient(loss_tr, model.trainable_variables)
  optimizer.apply(gradients, model.trainable_variables)
  return outputs_tr, loss_tr

def test_step(input_graph_tr, target_tr, num_processing_steps_tr=1):
  with tf.GradientTape() as tape:
    outputs_tr = model(input_graph_tr, num_processing_steps_tr)
    loss_tr = create_loss(target_tr, outputs_tr)
    loss_tr = tf.math.reduce_sum(loss_tr) / num_processing_steps_tr
  # gradients = tape.gradient(loss_tr, model.trainable_variables)
  # optimizer.apply(gradients, model.trainable_variables)
  return outputs_tr, loss_tr

def get_edge_index(r,s):
  edges=[]
  for e in range(0,len(r)):
    edges.append(np.array([r[e],s[e]]))
  return edges

graph_tuple = GenerateGraphTuple()

num_training_iterations = 5000

last_iteration = 0
last_iteration = 0

losses_tr = []

logged_iterations = []
losses_tr = []
corrects_tr = []
solveds_tr = []


# Get some example data that resembles the tensors that will be fed
# into update_step():
example_input_data, example_target_data = get_data(graph_tuple, batch_num=[0,40])[:2]
# Get the input signature for that function by obtaining the specs



input_signature = [
  utils_tf.specs_from_graphs_tuple(example_input_data),
  tf.TensorSpec(example_target_data.nodes.shape)
]

# Compile the update function using the input signature for speedy code.
compiled_update_step = tf.function(update_step, input_signature=input_signature)
graph = GenerateGraphVisual()
compiled_test_step = tf.function(test_step, input_signature=input_signature)

log_every_seconds = 10

start_time = time.time()
last_log_time = start_time


pcd = o3d.geometry.PointCloud()
for iteration in range(last_iteration, num_training_iterations):
  last_iteration = iteration
  input_graph_tr , target_tr, base_graph_tr = get_data(graph_tuple, batch_num =[0,40])
  output_tr, loss_tr = compiled_update_step(input_graph_tr, target_tr.nodes)

  the_time = time.time()
  elapsed_since_last_log = the_time - last_log_time
  if elapsed_since_last_log > log_every_seconds:
    last_log_time = the_time
    # Replace the globals again to prevent exceptions.
    output_tr[-1] = output_tr[-1].replace(globals=None)
    elapsed = time.time() - start_time
    losses_tr.append(loss_tr.numpy())
    logged_iterations.append(iteration)
    print("# {:05d}, T {:.1f}, Ltr {:.4f}".format(iteration, elapsed, loss_tr.numpy()))

############# test data
last_test_iter = 0
start_time_test = time.time()
last_test_log_time = start_time
num_test_iterations = 100
losses_test=[]
logged_test_iterations=[]
log_test_every_seconds = 1
for curr_iter in range(last_test_iter, num_test_iterations):
  last_test_iter = curr_iter
  input_graph_test , target_test, base_graph_test = get_data(graph_tuple, batch_num =[40,50])
  output_test, loss_test = compiled_test_step(input_graph_test, target_test.nodes)

  the_time_test = time.time()
  elapsed_since_last_log_test = the_time_test - last_test_log_time
  if elapsed_since_last_log_test > log_test_every_seconds:
    last_log_time_test = the_time
    # Replace the globals again to prevent exceptions.
    output_test[-1] = output_test[-1].replace(globals=None)
    elapsed_test = time.time() - start_time_test
    losses_test.append(loss_test.numpy())
    logged_test_iterations.append(curr_iter)
    print("# {:05d}, T {:.1f}, Ltr {:.4f}".format(curr_iter, elapsed_test, loss_test.numpy()))



graph.visualize_points(pos=np.asarray(np.array(target_tr[0])))


edges=get_edge_index(np.array(output_tr[0].receivers), np.array(output_tr[0].senders))
data = graph.generate_graph_data(np.asarray(np.array(output_tr[0].nodes)), edges)


graph.visualize_points(pos=data.pos)
  
graph.visualize_points(pos=data.pos, edge_index = data.edge_index)









# Plot results curves.
fig = plt.figure(11, figsize=(18, 3))
fig.clf()
x = np.array(logged_iterations)
# Loss.
y_tr = losses_tr
# y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="Training")
# ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across training")
ax.set_xlabel("Training iteration")
ax.set_ylabel("next step loss (num_process = 5, target_t=1)")
ax.legend()
plt.show()

# Plot results curves.
fig = plt.figure(11, figsize=(18, 3))
fig.clf()
x = np.array(logged_test_iterations)
# Loss.
y_tr = losses_test
# y_ge = losses_ge
ax = fig.add_subplot(1, 3, 1)
ax.plot(x, y_tr, "k", label="test")
# ax.plot(x, y_ge, "k--", label="Test/generalization")
ax.set_title("Loss across test")
ax.set_xlabel("test iteration")
ax.set_ylabel("next step loss (num_process = 5, target_t=1)")
ax.legend()
plt.show()    