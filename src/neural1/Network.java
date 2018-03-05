package neural1;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

//TODO user Array instead of List for performance
public class Network implements Serializable {
	private static final long serialVersionUID = 3608908257130684121L;

	private List<List<Neuron>> networkLayer;
	private List<double[][]> networkEdges;
	private int in;
	private int out;
	private int hiddenLayer;
	private double epsilon = 0.1; // 0.1 means 10% difference is ok for training
	private double learningrate = 0.0001;
	private static final double MIN=0.0;
	private static final double MAX=1.0;
	public boolean printResult=false;
	
	public void print(){
		DecimalFormat df = new DecimalFormat("0.000");
		int k=0;
		for(List<Neuron> layer:networkLayer){
			for(Neuron n:layer){
				System.out.print("(in: "+df.format(n.getInput())+" out: "+df.format(n.getOutput())+" )\t");
			}
			if(k<networkEdges.size()){
				double[][] weights=networkEdges.get(k);
		
			System.out.print("\n\n");
			for(int i=0;i<weights.length;i++){
				System.out.print("["+i+"]\t");
				for(int j=0;j<weights[i].length;j++){
					System.out.print(df.format(weights[i][j])+" ");
				}
				
			}
			System.out.print("\n\n");}
			k++;
		}
	}
	public Network(int in, int hiddenLayer, int nodesPerLayer, int out) {
		networkLayer = new ArrayList<>();
		networkEdges = new ArrayList<>();
		this.in = in;
		this.hiddenLayer = hiddenLayer;
		this.out = out;

		// build empty network, new neurons and weight matrix with w=1 for each
		// edge

		// 1. input Layer
		List<Neuron> inputLayer = new ArrayList<>();
		for (int i = 0; i < in; i++) {
			inputLayer.add(new Neuron());
		}
		networkLayer.add(0, inputLayer);

		// 2. hidden Layer
		for (int i = 0; i < hiddenLayer; i++) {
			List<Neuron> hLayer = new ArrayList<>();
			for (int j = 0; j < nodesPerLayer; j++) {
				hLayer.add(new Neuron());
			}
			networkLayer.add(1 + i, hLayer);
		}

		// 3. output Layer
		List<Neuron> outputLayer = new ArrayList<>();
		for (int i = 0; i < out; i++) {
			outputLayer.add(new Neuron());
		}
		networkLayer.add(networkLayer.size(), outputLayer);

		// 4. weight matrix
		networkEdges.add(addWeightMatrix(in, nodesPerLayer));
		for (int i = 1; i < hiddenLayer; i++) {
			networkEdges.add(addWeightMatrix(nodesPerLayer, nodesPerLayer));
		}
		networkEdges.add(addWeightMatrix(nodesPerLayer, out));
		
	}

	private double[][] addWeightMatrix(int in, int out) {
		double[][] matrix = new double[in][out];
		for (int i = 0; i < in; i++) {
			for (int j = 0; j < out; j++) {
				matrix[i][j] = getInitialWieght();
			}
		}
		return matrix;
	}

	public double[] process(double[] input) {
		if (input.length > in) {
			throw new RuntimeException();
		}
		resetInputs();
		// stimulate the input layer neurons
		for (int i = 0; i < input.length; i++) {
			this.networkLayer.get(0).get(i).stimulate(input[i]);
		}

		// let the neurons fire forward
		for (int i = 0; i < this.networkLayer.size() - 1; i++) {
			List<Neuron> lastLayer = networkLayer.get(i);
			List<Neuron> nextLayer = networkLayer.get(i+1);
			double[][] weights = networkEdges.get(i);
			for (int j = 0; j < lastLayer.size(); j++) {
				for (int k = 0; k < nextLayer.size(); k++) {
					// fire
					nextLayer.get(k).stimulate(lastLayer.get(j).getOutput() * weights[j][k]);
				}
			}
		}

		// fetch the output layer
		double[] out = new double[this.out];
		DecimalFormat df = new DecimalFormat("0.00");
//		System.out.println("Result: ...");
		for (int i = 0; i < networkLayer.get(networkLayer.size() - 1).size(); i++) {

			out[i] = networkLayer.get(networkLayer.size() - 1).get(i).getOutput();
			if(printResult){
			System.out.println(i+" : "+df.format(networkLayer.get(networkLayer.size() - 1).get(i).getOutput()));
			}
		}	
		return out;

	}

	public void train(double[] input, double[] expectedOut) {
		double[] out = process(input);

		// back propagation for Output layer
		List<Neuron> outputLayer = networkLayer.get(networkLayer.size() - 1);
		double[][] weights = networkEdges.get(networkEdges.size() - 1);
		for (int i = 0; i < outputLayer.size(); i++) {
			// get Failure
			double failure = outputLayer.get(i).getFailure(expectedOut[i]);
			// calc new weight
			for (int j = 0; j < weights.length; j++) {
				double deltaW = failure * outputLayer.get(i).getInput() * learningrate;
				double old=weights[j][i];
				if((old + deltaW)>=MIN && (old + deltaW)<=MAX){
				weights[j][i] = old + deltaW; //  + or - ???
				}
//				System.out.println("Layer 0 Neuron "+i+ " Gewicht "+j+" failure "+failure+" new weight: "+	weights[j][i] );												// - ???
				
			}
		}

		// back propagation for Hidden & Input layer
		for (int i = networkLayer.size() - 2; i > 0; i--) {
			List<Neuron> currentLayer = networkLayer.get(i);
			List<Neuron> nextLayer = networkLayer.get(i + 1);

			double[][] currWeights = networkEdges.get(i - 1);
			double[][] nextWeights = networkEdges.get(i);

			for (int j = 0; j < currentLayer.size(); j++) {
				// get Failure
				double prevfailure = 0;
				for (int k = 0; k < nextLayer.size() - 1; k++) {
					prevfailure += nextWeights[j][k] * nextLayer.get(k).failure;
				}
				double failure = currentLayer.get(j).getHiddenFailure(prevfailure);
				// calc new weight
				for (int k = 0; k < currWeights.length; k++) {
					double deltaW = failure * currentLayer.get(j).getInput() * learningrate;
					double old=currWeights[k][j] ;
					if((old + deltaW)>=MIN && (old + deltaW)<=MAX){
					currWeights[k][j] = old + deltaW; // TODO + or
					}
//					System.out.println("Layer "+i+" Neuron "+j+ " Gewicht "+k+" prevfailure "+prevfailure+" failure "+failure+" new weight: "+	currWeights[k][j] );												// - ???
				}
			}
		}
	}
	
	public void resetInputs(){
		for(List<Neuron> l:networkLayer){
			for(Neuron n:l){
				n.reset();
			}
		}
	}
	
	public double getInitialWieght(){
		return Math.random();
	}

}
