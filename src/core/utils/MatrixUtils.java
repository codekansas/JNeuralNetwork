package core.utils;

import org.ejml.simple.SimpleMatrix;

public class MatrixUtils {
	public static SimpleMatrix getColumnMatrix(double... data) {
		SimpleMatrix m = new SimpleMatrix(1, data.length);
		m.setRow(0, 0, data);
		return m;
	}
}
