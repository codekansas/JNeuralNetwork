package core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.ejml.simple.SimpleMatrix;

import core.objectives.MeanSquaredError;
import core.objectives.ObjectiveFunction;

public class Model {
	private List<Layer> layers;
	private boolean built;

	private int n_in, n_out, batch_size;
	private double learning_rate;
	private ObjectiveFunction objective;
	
	public Model(int n_in, int n_out) {
		this(n_in, n_out, 10, 0.1, new MeanSquaredError());
	}
	
	public Model(int n_in, int n_out, int batch_size, double learning_rate, ObjectiveFunction objective) {
		layers = new ArrayList<>();
		built = false;
		
		this.n_in = n_in;
		this.n_out = n_out;
		this.batch_size = batch_size;
		this.learning_rate = learning_rate;
		this.objective = objective;
	}
	
	public void add(Layer l) {
		layers.add(l);
	}
	
	protected void build() {
		int n_dim = this.n_in;
		
		for (Layer l : layers) {
			n_dim = l.build(n_dim);
		}
		
		if (n_dim != this.n_out) {
			throw new ModelException(String.format("Required %d output dimensions, last layer has %d", this.n_out, n_dim));
		}
		
		built = true;
	}
	
	public List<SimpleMatrix> evaluate(List<SimpleMatrix> inputs) {
		if (!built) {
			build();
		}
		
		List<SimpleMatrix> outputs = new ArrayList<>(inputs.size());
		for (SimpleMatrix input : inputs) {
			for (int i = 0; i < layers.size(); i++) {
				input = layers.get(i).evaluate(input);
			}
			outputs.add(input);
		}
		return outputs;
	}
	
	public double error(List<SimpleMatrix> inputs, List<SimpleMatrix> targets) {
		return objective.evaluate(evaluate(inputs), targets);
	}
	
	public void train(List<SimpleMatrix> inputs, List<SimpleMatrix> targets, int nb_epochs) {
		if (!built) {
			build();
		}
		
		if (inputs.size() != targets.size()) {
			throw new ModelException(String.format("Number of input and output sequences must be equal (recieved %d inputs, %d outputs)", inputs.size(), targets.size()));
		}
		
		for (int epoch = 0; epoch < nb_epochs; epoch++) {
			for (int i = 0; i < inputs.size(); i += batch_size) {
				runBatch(inputs.subList(i, Math.min(i + batch_size, inputs.size())),
						targets.subList(i, Math.min(i + batch_size, inputs.size())));
			}
		}
	}
	
	protected void runBatch(List<SimpleMatrix> inputs, List<SimpleMatrix> targets) {
		List<List<SimpleMatrix>> batch = new ArrayList<>();
		
		// Forward step
		for (SimpleMatrix input : inputs) {
			List<SimpleMatrix> outputs = new ArrayList<>();
			for (int i = 0; i < layers.size(); i++) {
				outputs.add(input);
				input = layers.get(i).evaluate(input);
			}
			outputs.add(input);
			batch.add(outputs);
		}

		// Backprop step
		List<List<Map<String,SimpleMatrix>>> updates = new ArrayList<>(inputs.size());
		for (int i = 0; i < batch.size(); i++) {
			List<SimpleMatrix> outputs = batch.get(i);
			SimpleMatrix gradient = objective.gradient(outputs.get(outputs.size()-1), targets.get(i));
			List<Map<String,SimpleMatrix>> update = new ArrayList<>();
			for (int j = layers.size()-1; j >= 0; j--) {
				Map<String,SimpleMatrix> backprop = layers.get(j).backpropagate(gradient, outputs.get(j), outputs.get(j+1));
				gradient = backprop.get("propError");
				update.add(backprop);
			}
			Collections.reverse(update);
			updates.add(update);
		}
		
		// Update step
		for (int j = 0; j < layers.size(); j++) {
			for (int i = 0; i < batch.size(); i++) {
				layers.get(j).update(updates.get(i).get(j), learning_rate);
			}
		}
	}

	public class ModelException extends RuntimeException {
		private static final long serialVersionUID = 420L;

		public ModelException(String message) {
			super(message);
		}
	}
}