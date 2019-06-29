Simple Neural Network Library (snn)
===================================

This project is my attempt to create a simple neural network
framework in C++ (cpp17) that is powerful enough to give decent
results in [MNIST dataset](http://yann.lecun.com/exdb/mnist/) in decent running time.

The project is inspired by the simplicity of Keras. e.g. creating a simple network for MNIST:

```cpp
    models::sequential m;
    m.add("dense(units=300, input=784)");
    m.add("dense(units=10)");
    m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.5)");
    m.summary();
```