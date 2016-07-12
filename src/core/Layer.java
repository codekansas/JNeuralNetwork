package core;

import java.util.Map;

import org.ejml.simple.SimpleMatrix;

public abstract class Layer {
	public abstract int build(int n_in);
	public abstract SimpleMatrix evaluate(SimpleMatrix input);
	public abstract Map<String,SimpleMatrix> backpropagate(SimpleMatrix error, SimpleMatrix input, SimpleMatrix output);
	public abstract void update(Map<String,SimpleMatrix> outputs, double learning_rate);
}
