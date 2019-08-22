package com.a1arick.my_nn;

public class Layer {
    private Layer predLayer;
    private Layer nextLayer;
    private double supportWeight;
    private double[][] weights;
    private double[] delta;
    private double[] neurons;
    private int input;
    private int output;
    private Activation activation;

    public Layer(int n, Activation activation) {
        neurons = new double[n];
        input = n;
        this.activation = activation;
    }

    public Layer(Layer predLayer, int n, Activation activation) {
        this.predLayer = predLayer;
        predLayer.setNextLayer(this);
        predLayer.setOutput(n);
        predLayer.getNeuronsNextLayer();
        input = n;
        this.activation = activation;
    }

    public double[] getDelta() {
        return delta;
    }

    public void calcDeltaLastLayer(double[] answer) {
        delta = new double[neurons.length];
        for (int i = 0; i < answer.length; i++) {
            delta[i] = answer[i] - neurons[i];
            //System.out.println(delta[i]);
        }
    }

    public void calcDelta() {
        double[] delta1 = nextLayer.getDelta();
        delta = new double[neurons.length];
        for (int j = 0; j < output; j++) {
            for (int i = 0; i < input; i++) {
                delta[i] += delta1[j] * weights[i][j];
            }
        }
    }

    public void updateWeights(double alpha) {
        double[] nextLayerDelta = nextLayer.getDelta();
        double[] nextLayerNeurons = nextLayer.getNeurons();
        for (int j = 0; j < output; j++) {
            for (int i = 0; i < input; i++) {
                weights[i][j] = weights[i][j] + alpha*nextLayerDelta[j]*neurons[i]*activation.derivative(nextLayerNeurons[j]);
            }
        }
    }

    public double[] getNeurons() {
        return neurons;
    }

    public void getNeuronsNextLayer() {
        double[] nextNeurons = new double[output];
        for (int j = 0; j < output; j++) {
            double sum = supportWeight;
            for (int i = 0; i < input; i++) {
                sum += weights[i][j] * neurons[i];
            }
            //nextNeurons[j] = 1 / (1 + Math.exp(-sum));
            nextNeurons[j] = activation.f(sum);
        }
        nextLayer.setNeurons(nextNeurons);
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void setNeurons(double[] neurons) {
        this.neurons = neurons;
    }

    public void setOutput(int output) {
        this.output = output;
        initWeight();
    }

    private void initWeight() {
        weights = new double[input][output];
        for (int i = 0; i < input; i++) {
            for (int j = 0; j < output; j++) {
                weights[i][j] = Math.random() - 0.5;
            }
        }
    }
}
