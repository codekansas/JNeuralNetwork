package core.initializers;

import java.util.Random;

import org.ejml.simple.SimpleMatrix;

public class GlorotInitializer extends Initializer {

	@Override
	public SimpleMatrix initialize(int n_in, int n_out) {
		double range = 4. * Math.sqrt(6. / (n_in + n_out));
		return SimpleMatrix.random(n_in, n_out, -range, range, new Random(42));
	}

}
