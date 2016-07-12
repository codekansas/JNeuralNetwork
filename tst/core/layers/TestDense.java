package core.layers;

import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

import core.Model;
import core.objectives.MeanSquaredError;
import core.utils.MatrixUtils;

public class TestDense {
	
	@Test
	public void testDimensions() {
		int n_in = 2, n_out = 3;
		
		Dense d = new Dense(n_out);
		d.build(n_in);
		
		SimpleMatrix input = SimpleMatrix.random(1, n_in, -1, 1, new Random(42));
		SimpleMatrix output = d.evaluate(input);
		
		assertEquals(output.numRows(), 1);
		assertEquals(output.numCols(), n_out);
	}
	
	@Test
	public void testLearning() {
		int n_in = 2, n_out = 1;
		
		Model m = new Model(n_in, n_out, 4, 0.1, new MeanSquaredError());
		m.add(new Dense(5));
		m.add(new Dense(1));
		
		List<SimpleMatrix> inputs = Arrays.asList(new SimpleMatrix[] {
			MatrixUtils.getColumnMatrix(1.0, 0.0),
			MatrixUtils.getColumnMatrix(0.0, 1.0),
			MatrixUtils.getColumnMatrix(1.0, 1.0),
			MatrixUtils.getColumnMatrix(0.0, 0.0),
		});
		
		List<SimpleMatrix> targets = Arrays.asList(new SimpleMatrix[] {
			MatrixUtils.getColumnMatrix(0.0),
			MatrixUtils.getColumnMatrix(0.0),
			MatrixUtils.getColumnMatrix(1.0),
			MatrixUtils.getColumnMatrix(1.0),
		});
		
		double errorBefore = m.error(inputs, targets);
		m.train(inputs, targets, 60000);
		double errorAfter = m.error(inputs, targets);
		
//		System.out.println(errorBefore);
//		System.out.println(errorAfter);
		
		assertTrue(errorBefore > errorAfter);
	}
}
