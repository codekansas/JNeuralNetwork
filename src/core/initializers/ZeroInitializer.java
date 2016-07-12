package core.initializers;

import org.ejml.simple.SimpleMatrix;

public class ZeroInitializer extends Initializer {

	@Override
	public SimpleMatrix initialize(int n_in, int n_out) {
		return new SimpleMatrix(n_in, n_out);
	}

}
