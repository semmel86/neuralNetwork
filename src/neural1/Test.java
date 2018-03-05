package neural1;

import java.util.ArrayList;
import java.util.List;

public class Test {

	private static Network neuralNetwork;

	public static void main(String[] args) {
		// new Network
		neuralNetwork = new Network(3, 2, 5, 3);

		double[][] inputs = { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 10.0 } };
		double[][] outputs = { { 1.0, 0.0 ,0.3}, { 1.0, 0.0 ,0.3}, { 0.0, 0.0 ,0.3 } };

		long time=System.currentTimeMillis();
		int n=1000000;
		for (int i = 0; i < n; i++) {
			neuralNetwork.train(inputs[2], outputs[2]);
			neuralNetwork.train(inputs[0], outputs[0]);
			neuralNetwork.train(inputs[1], outputs[1]);
//			neuralNetwork.train(inputs[2], outputs[2]);
			System.out.println( (((double)i) / n)*100 +" %");
		}
		System.out.println("Finished after: "+ (System.currentTimeMillis()-time)/1000 +" sek");
//		double[] result = neuralNetwork.process(inputs[2]);
//		for (double d : result) {
//			System.out.println(d);
//		}
		neuralNetwork.printResult=true;
		neuralNetwork.process(inputs[0]);
//		neuralNetwork.print();
		neuralNetwork.process(inputs[1]);
//		neuralNetwork.print();
		neuralNetwork.process(inputs[2]);
//		neuralNetwork.print();
		
	}

}
