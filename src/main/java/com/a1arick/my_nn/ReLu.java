package com.a1arick.my_nn;

public class ReLu implements Activation {
    public double f(double sum) {
        if(sum >= 0) return sum;
        else return 0;
    }

    public double derivative(double sum) {
        if(sum >= 0) return 1;
        else return 0;
    }
}
