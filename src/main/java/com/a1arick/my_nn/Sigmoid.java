package com.a1arick.my_nn;

public class Sigmoid implements Activation {
    public double f(double sum) {
        return 1 / (1 + Math.exp(-sum));
    }

    public double derivative(double sum) {
        return f(sum) * (1 - f(sum));
    }
}
