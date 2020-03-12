import sys
import os
import io
import matplotlib.pyplot as plt
import tensorflow as tf

def readFile(filename):
	print("Reading file - " + filename)
	with open(filename) as f:
		data = f.read()
		lst = data[1:-1].split(", ")
	return [float(x) for x in lst]

def create_graph(arr, title):
    def gen_plot(arr):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.plot(arr)
        plt.title(str(title))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

    print("Creating graph for - " + title)
    plot_buf = gen_plot(arr)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(plot_buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)

    # Add image summary
    summary_op = tf.summary.image("plot", image)

    # Session
    with tf.Session() as sess:
        # Run
        summary = sess.run(summary_op)
        # Write summary
        writer = tf.summary.FileWriter("graphs")
        writer.add_summary(summary)
        writer.close()

if __name__ == '__main__':
	path = sys.argv[1]
	arr = []
	if os.path.isdir(path): 
		for filename in os.listdir(path):
			arr = readFile(path + filename)
			create_graph(arr, filename)
	else:
		arr = readFile(path)
		create_graph(arr, path)
	
	print("\nFinished creating the graph(s). View them by running tensorboard on the './graphs/' folder.")

