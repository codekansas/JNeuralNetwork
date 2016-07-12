package core.layers;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

import core.Layer;
import core.activations.ActivationFunction;
import core.activations.SigmoidActivation;
import core.initializers.GlorotInitializer;
import core.initializers.Initializer;
import core.initializers.ZeroInitializer;

public class Dense extends Layer {
	
	protected SimpleMatrix weights, biases;
	
	protected int n_out;
	protected ActivationFunction activation;
	protected Initializer weightInitializer, biasInitializer;
	
	public Dense(int n_out) {
		this(n_out, new SigmoidActivation(), new GlorotInitializer(), new ZeroInitializer());
	}
	
	public Dense(int n_out, ActivationFunction activation, Initializer weightInitializer, Initializer biasInitializer) {
		this.n_out = n_out;
		this.activation = activation;
		this.weightInitializer = weightInitializer;
		this.biasInitializer = biasInitializer;
	}

	@Override
	public int build(int n_in) {
		Random seed = new Random(42);
		weights = SimpleMatrix.random(n_in, n_out, -1, 1, seed);
		biases = SimpleMatrix.random(1, n_out, -1, 1, seed);
		return n_out;
	}

	@Override
	public SimpleMatrix evaluate(SimpleMatrix input) {
		SimpleMatrix output = activation.evaluate(input.mult(weights).plus(biases));
		return output;
	}

	@Override
	public Map<String,SimpleMatrix> backpropagate(SimpleMatrix error, SimpleMatrix input, SimpleMatrix output) {
		Map<String,SimpleMatrix> model = new HashMap<>();
		SimpleMatrix delta = error.elementMult(activation.deriv(output));
		model.put("biasUpdates", delta);
		model.put("weightUpdates", input.transpose().mult(delta));
		model.put("propError", delta.mult(weights.transpose()));
		return model;
	}

	@Override
	public void update(Map<String, SimpleMatrix> outputs, double learning_rate) {
		weights = weights.plus(outputs.get("weightUpdates").divide(1 / learning_rate));
		biases = biases.plus(outputs.get("biasUpdates").divide(1 / learning_rate));
	}
	
}
