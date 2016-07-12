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
	
	public static String ACTIVATION = "activation";
	public static String WEIGHT_INITIALIZER = "weightInitializer";
	public static String BIAS_INITIALIZER = "biasInitializer";
	
	public Dense(int n_out) {
		this(n_out, new HashMap<String,Object>());
	}
	
	public Dense(int n_out, Map<String,Object> params) {
		this.n_out = n_out;
		activation = (ActivationFunction) params.getOrDefault(ACTIVATION, new SigmoidActivation());
		weightInitializer = (Initializer) params.getOrDefault(WEIGHT_INITIALIZER, new GlorotInitializer());
		biasInitializer = (Initializer) params.getOrDefault(BIAS_INITIALIZER, new ZeroInitializer());
	}

	@Override
	public int build(int n_in) {
		weights = weightInitializer.initialize(n_in, n_out);
		biases = biasInitializer.initialize(1, n_out);
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
