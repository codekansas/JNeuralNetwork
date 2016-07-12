package core.initializers;

import org.ejml.simple.SimpleMatrix;

public abstract class Initializer {
	public abstract SimpleMatrix initialize(int n_in, int n_out);
}
