# Hi! If you've landed on this file, chances are:
# I've directed you, or you've been snooping around my GitHub. In any case, welcome!
# In this basics file, I'll be setting up TensorFlow and a simple graph computation.

# %% Import TensorFlow and pyplot
import tensorflow as tf
import matplotlib.pyplot as plt

# By default, there is a graph: tf.get_default_graph()
# and any new operations are added to this graph.
# The result of a tf.Operation is a tf.Tensor, which holds
# the values.

# %% Creating a tensor
n_values = 32
x = tf.linspace(-3.0, 3.0, n_values)

# %% Construct a session to execute the graph
sess = tf.Session()
result = sess.run(x)

# %% Pass a session to the eval fn:
x.eval(session=sess)
# x.eval() does not work alone; it requires a session

sess.close()
sess = tf.InteractiveSession()

# %% Now that we've added a session, x.eval() will work 🤠
x.eval()

# %% Now a tf.Operation
# Using the values from above -> [-3, 3] to create a Gaussian Distribution
sigma = 1.0
mean = 0.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                   (2.0 * tf.pow(sigma, 2.0)))) *
     (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

# %% By default, new operations are added to the default Graph
assert z.graph is tf.get_default_graph()

# %% Execute the graph and plot the result
plt.plot(z.eval())

# %% We can find out the shape of a tensor like so:
print(z.get_shape())

# %% Or in a more friendly format
print(z.get_shape().as_list())


#  tf.shape fn, which will returns a tensor which can be
#  eval'ed, rather than a discrete value of tf.Dimension
#  value of tf.Dimension
print(tf.shape(z).eval())

# %% Combine tensors
print(tf.stack([tf.shape(z), tf.shape(z), [3], [4]]).eval())

# %% Multiply the two to get a 2d gaussian
z_2d = tf.matmul(tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))

# %% Execute the graph and store the value that `out` represents in `result`
plt.imshow(z_2d.eval())

# %% Create a gabor patch:
x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
y = tf.reshape(tf.ones_like(x), [1, n_values])
z = tf.multiply(tf.matmul(x, y), z_2d)
plt.imshow(z.eval())

# %% List all the operations of a graph:
ops = tf.get_default_graph().get_operations()
print([op.name for op in ops])

def gabor(n_values=32, sigma=1.0, mean=0.0):
    x = tf.linspace(-3.0, 3.0, n_values)
    z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) /
                       (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
    gauss_kernel = tf.matmul(
        tf.reshape(z, [n_values, 1]), tf.reshape(z, [1, n_values]))
    x = tf.reshape(tf.sin(tf.linspace(-3.0, 3.0, n_values)), [n_values, 1])
    y = tf.reshape(tf.ones_like(x), [1, n_values])
    gabor_kernel = tf.multiply(tf.matmul(x, y), gauss_kernel)
    return gabor_kernel

plt.imshow(gabor().eval())

# %% Convolve
def convolve(img, W):

    if len(W.get_shape()) == 2:
        dims = W.get_shape().as_list() + [1, 1]
        W = tf.reshape(W, dims)

    if len(img.get_shape()) == 2:
        # num x height x width x channels
        dims = [1] + img.get_shape().as_list() + [1]
        img = tf.reshape(img, dims)
    elif len(img.get_shape()) == 3:
        dims = [1] + img.get_shape().as_list()
        img = tf.reshape(img, dims)
        # if the image is 3 channels, then our convolution
        # kernel needs to be repeated for each input channel
        W = tf.concat(axis=2, values=[W, W, W])

    convolved = tf.nn.conv2d(img, W,
                             strides=[1, 1, 1, 1], padding='SAME')
    return convolved

# %% Load up an image:
from skimage import data
img = data.astronaut()
plt.imshow(img)
print(img.shape)

x = tf.placeholder(tf.float32, shape=img.shape)

out = convolve(x, gabor())

# %% Send the image into the graph and compute the result
result = tf.squeeze(out).eval(feed_dict={x: img})
plt.imshow(result)
©
