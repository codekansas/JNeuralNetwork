package core.objectives;

import java.util.ArrayList;
import java.util.List;

import org.ejml.simple.SimpleMatrix;

public abstract class ObjectiveFunction {
	protected void assertValid(List<SimpleMatrix> outputs, List<SimpleMatrix> targets) {
		if (outputs.size() != targets.size()) {
			throw new ObjectiveFunctionException(
					String.format("Must provide the same number of outputs and targets (recieved %d outputs, %d targets)",
							outputs.size(), targets.size()));
		}
		
		for (int i = 0; i < outputs.size(); i++) {
			if (outputs.get(i).getNumElements() != targets.get(i).getNumElements()) {
				throw new ObjectiveFunctionException(
						String.format("Output and target shape are not the same (recieved %d for output, %d for target)",
								outputs.get(i).getNumElements(), targets.get(i).getNumElements()));
			}
		}
	}
	
	public abstract double evaluate(SimpleMatrix outputs, SimpleMatrix targets);
	public abstract SimpleMatrix gradient(SimpleMatrix outputs, SimpleMatrix targets);
	
	public double evaluate(List<SimpleMatrix> outputs, List<SimpleMatrix> targets) {
		assertValid(outputs, targets);
		
		double error = 0.;
		for (int i = 0; i < outputs.size(); i++) {
			error += evaluate(outputs.get(i).copy(), targets.get(i));
		}
		
		error /= (double) outputs.size();
		
		return error;
	}
	
	public List<SimpleMatrix> gradient(List<SimpleMatrix> outputs, List<SimpleMatrix> targets) {
		assertValid(outputs, targets);
		
		List<SimpleMatrix> gradients = new ArrayList<>(outputs.size());
		for (int i = 0; i < outputs.size(); i++) {
			gradients.add(gradient(outputs.get(i).copy(), targets.get(i)).divide(outputs.size()));
		}
		
		return gradients;
	}
	
	public static class ObjectiveFunctionException extends RuntimeException {
		private static final long serialVersionUID = 69L;

		public ObjectiveFunctionException(String message) {
			super(message);
		}
	}
}
