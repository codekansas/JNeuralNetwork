package core.objectives;

import org.ejml.simple.SimpleMatrix;

public class MeanSquaredError extends ObjectiveFunction {

	@Override
	public double evaluate(SimpleMatrix output, SimpleMatrix target) {
		SimpleMatrix diff = output.minus(target);
		return diff.dot(diff) / 2.;
	}
	
	@Override
	public SimpleMatrix gradient(SimpleMatrix output, SimpleMatrix target) {
		return target.minus(output);
	}

}
