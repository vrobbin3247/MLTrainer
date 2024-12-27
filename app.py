from mlp import Neuron,Value,learn
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, request, render_template, redirect, url_for


app = Flask(__name__)

# Function to plot loss and return as an image
def plot_loss(log):
    steps = [entry[0] for entry in log]
    losses = [entry[1] for entry in log]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, marker='o', linestyle='-', color='b', label='Loss')
    plt.title('Training Loss Over Steps')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get inputs from the form
        xs_raw = request.form.get("xs")
        ys_raw = request.form.get("ys")
        layer_sizes_raw = request.form.get("layers")
        steps = int(request.form.get("steps"))
        learning_rate = float(request.form.get("learning_rate"))

        # Parse xs and ys
        xs = [[Value.Value(float(v), label='input') for v in sample.split(",")] for sample in xs_raw.split(";")]
        ys = [float(y) for y in ys_raw.split(",")]

        # Parse layer sizes
        layer_sizes = [int(size) for size in layer_sizes_raw.split(",")]
        n = Neuron.MLP(len(xs[0]), layer_sizes + [1])  # Add output layer of size 1

        # Train the model
        log = learn.learn(steps, learning_rate, xs, ys, n)

        # Generate the plot
        plot_url = plot_loss(log)

        # Redirect to results page with plot
        return render_template("results.html", plot_url=plot_url)

    return render_template("index.html")

@app.route("/about")
def about():
    return "<h1>About the MLP Trainer</h1><p>This is a simple Flask app for training an MLP and visualizing loss.</p>"

if __name__ == "__main__":
    app.run(debug=True)