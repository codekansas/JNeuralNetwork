package core.activations;

import org.ejml.simple.SimpleMatrix;

public class SigmoidActivation extends ActivationFunction {

	@Override
	public SimpleMatrix evaluate(SimpleMatrix input) {
		SimpleMatrix output = new SimpleMatrix(input.numRows(), input.numCols());
		output.set(1);
		return output.elementDiv(input.negative().elementExp().plus(1));
	}

	@Override
	public SimpleMatrix deriv(SimpleMatrix input) {
		return input.elementMult(input.minus(1).negative());
	}
}
