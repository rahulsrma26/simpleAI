Getting Started with MNIST (snn)
================================

 [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a good dataset for people who want to try different learning techniques on real-world data while spending minimal efforts on preprocessing and formatting.

You can download it manually from the [site](http://yann.lecun.com/exdb/mnist/) or use [download_mnist.py](../../scripts/download_mnist.py) to download the dataset.

```sh
mkdir data
python scripts/download_mnist.py data/
```

In this example we will write a simple single hidden layer neural network to recognise digits. You can lookup the source [here](../../examples/demo_mnist.cpp)

---

First we will include headers for model and dataset.
```cpp
#include "snn/snn.hpp"
#include "snn/dataset/mnist.hpp"
#include <iostream>
```

Load dataset from path prefixes.
```cpp
auto [trainX, trainY] = dataset::mnist::load(data_dir + "/train");
auto [testX, testY] = dataset::mnist::load(data_dir + "/t10k");
```

Flatten 2D images to 1D. Reshape function will reshape the tensor from Nx28x28 to Nx784. Here '0' is similar to '-1' in numpy's reshape.
```cpp
trainX.reshape({0, 784});
testX.reshape({0, 784});
```

Defining a seuential model.
```cpp
models::sequential m;
m.add("dense(units=300, input=784)");
m.add("dense(units=10)");
m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.5)");
m.summary();
```

Running for 20 epoches. 'run' function also shows the progress.
```cpp
const int epochs = 20;
for (int epoch = 1; epoch <= epochs; epoch++) {
    cout << "Epoch: " << epoch << '/' << epochs << endl;
    m.run(trainX, trainY, "batch_size=32");
    m.run(testX, testY, "batch_size=128, train=false");
}
```

---