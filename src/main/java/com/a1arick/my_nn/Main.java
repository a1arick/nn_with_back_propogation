package com.a1arick.my_nn;

public class Main {
    public static void main(String[] args) {
        NN nn = new NN();

        nn.addLayer(1, new ReLu());
        nn.addLayer(10, new ReLu());
        nn.addLayer(1, new ReLu());

        double[][] trainInput = new double[10000][1];
        double[][] trainOutput = new double[10000][1];

        for (int i = 0; i < 10000; i++) {
            trainInput[i][0] = Math.random();
            trainOutput[i][0] = trainInput[i][0] / 2 ;
        }

        nn.setInputData(trainInput);
        nn.setTryAnswer(trainOutput);

        nn.fit(1000);

        double[] a1 = new double[1];
        a1[0] = 0.1;
        nn.show(a1);

    }
}
