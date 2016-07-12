package core.activations;

import org.ejml.simple.SimpleMatrix;

public abstract class ActivationFunction {
	public abstract SimpleMatrix evaluate(SimpleMatrix input);
	public abstract SimpleMatrix deriv(SimpleMatrix input);
	
	public static class ActivationFunctionException extends RuntimeException {
		private static final long serialVersionUID = 69L;

		public ActivationFunctionException(String message) {
			super(message);
		}
	}
}
