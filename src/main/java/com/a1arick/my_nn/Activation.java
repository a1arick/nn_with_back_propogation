package com.a1arick.my_nn;

public interface Activation {
    double f(double sum);
    double derivative(double sum);
}
