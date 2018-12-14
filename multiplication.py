s = tf.placeholder("float") # Create a symbolic variable 's'
t = tf.placeholder("float") # Create a symbolic variable 't'

y = tf.multiply(s, t) # multiply the symbolic variables

with tf.Session() as sess: # create a session to evaluate the symbolic expressions
    print("%f should equal 2.0" % sess.run(y, feed_dict={s: 1, t: 2})) # eval expressions with parameters for s and t
    print("%f should equal 9.0" % sess.run(y, feed_dict={s: 3, t: 3}))
