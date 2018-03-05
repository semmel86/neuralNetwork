package neural1;

public class Neuron {

	private double input;
	public double failure;
	
	public double getOutput(){
		return Math.tanh(input);
	}
	
	public void stimulate(double s){
		input+=s;
	}
	
	public double getFailure(double z){
		this.failure= (1.0/(Math.pow(Math.cosh(input), 2)))*(z-Math.tanh(input));
		return failure;
	}
	
	public double getHiddenFailure(double w){
//		this.failure= (1.0/(Math.pow(Math.cosh(input), 2)))*(w);
		return (1.0/(Math.pow(Math.cosh(input), 2)))*(w);
	}
	
	public double getInput(){
		return input;
	}
	
	public void reset(){
		this.input=0;
	}
	@Override
	public String toString(){
		return "Neuron Input="+input+" output"+ getOutput();	
		}
	
	public String print(){
		return "|in: "+input+" |\n|out: "+this.getOutput()+" |\n\n";
				
	}
}
