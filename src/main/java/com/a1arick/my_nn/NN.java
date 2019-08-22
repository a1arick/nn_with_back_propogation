package com.a1arick.my_nn;

public class NN {
    private Layer[] layers = new Layer[50];
    private int size = 0;
    private double[][] inputData;
    private double[][] tryAnswer;
    private double alpha = 0.001;

    public void setInputData(double[][] inputData) {
        this.inputData = inputData;
    }

    public void setTryAnswer(double[][] tryAnswer) {
        this.tryAnswer = tryAnswer;
    }

    public void addLayer(int n, Activation activation) {
        if (size == 0) {
            layers[0] = new Layer(n, activation);
        } else {
            layers[size] = new Layer(layers[size - 1], n, activation);
        }
        size++;
    }

    private void fitHelper() {
        int k = 0;
        for (double[] inputDatum : inputData) {
            double[] answer = tryAnswer[k];

            drive(inputDatum);

            layers[size - 1].calcDeltaLastLayer(answer);
            for (int i = size - 2; i >= 1; i--) {
                layers[i].calcDelta();
            }

            for (int i = 0; i < size - 1; i++) {
                layers[i].updateWeights(alpha);
            }

            k++;
        }
    }

    private void drive(double[] input) {
        int m = 0;
        layers[0].setNeurons(input);
        while(m < size -1) {
            layers[m].getNeuronsNextLayer();
            m++;
        }
    }

    public void fit(int epoch) {
        while(epoch > 0) {
            fitHelper();
            epoch--;
        }
    }

    public void show(double[] input) {
        drive(input);
        for (double neuron : layers[size-1].getNeurons()) {
            System.out.println(neuron);
        }
    }
}
