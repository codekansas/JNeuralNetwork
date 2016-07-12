package core.initializers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class UniformInitializer extends Initializer {

	@Override
	public SimpleMatrix initialize(int n_in, int n_out) {
		return SimpleMatrix.random(n_in, n_out, -1, 1, new Random(42));
	}

}
