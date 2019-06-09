Simple Neural Network Library (v2 alpha)
========================================

This project is my attempt to create a simple neural network
code in C++ (cpp17) that is powerful enough to give decent
results in [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
You can download the dataset manually or run [download_mnist.py](download_mnist.py).

The project is inspired by the simplicity of Keras. Creating a network for MNIST.

Code snippet:
```cpp
    models::sequential m;
    m.add("dense(units=300, input=784)");
    m.add("dense(units=10)");
    m.compile("loss=cross_entropy(), optimizer=sgd(learning_rate=0.5)");
    m.summary();
```

Console Output:
![console_run](docs/plots/console_run.gif)

See [demo_mnist.cpp](examples/demo_mnist.cpp) for full code.

---

*	[Features](#features)
*	[Results](#results)
*	[Project structure](#project-structure)
*   [Building Project](#building-project)

---

[](#features)
## Currently Supported Features

### 1. Save/Load Models

### 2. Layers
*	Dense Layer

### 3. Activators
*	sigmoid
*	tanh
*	relu

### 4. Loss Functions
*	quadratic
*	hillinger
*	cross-entropy

### 5. Variable Initializers
*	zeros
*	normal
*	xavier

### 6. Optimizers
*	SGD

---

[](#results)
## Results

Experiment:
* Dataset: MNIST
* Hidden Layers: 1 Dense Layer (300 neurons)
* Learning Rate: 0.5 (Decay = 0)
* Batch Size: 32

See [demo_mnist.cpp](examples/demo_mnist.cpp) for full code.

Here is a comparision of different Loss-Functions (test-accuracy)
![alt text](docs/plots/loss_functions.svg)

---

[](#project-structure)
add soon...

---

[](#building-project)
## Building Project

### Building examples

```sh
git clone https://github.com/rahulsrma26/simpleNN.git
cd simpleNN
mkdir build
cd build
```

* Linux and MacOS (make)
    ```sh
    cmake ..
    make
    ```

* Windows

    Visual Studio 2017 and later can directly open a CMake project. However, if you want to build using Developer Command prompt and have [cmake](https://cmake.org/) installed then run:
    ```sh
    cmake ..
    msbuild simpleNN.sln /t:Build /p:Configuration=Release
    ```

### Running examples

```sh
# After build first download MNIST dataset
python ../download_mnist.py ../data
# mnist example
bin/demo_mnist ../data
# spiral example
bin/spiral_test
# mnist model with save/restore
bin/mnist_model
```
For windows users that are using MSVC check '.exe' files in bin\Release folder.

### Even More ways to build

You can also be build using [ninja](https://ninja-build.org/) across any platform.

* Using ninja
    ```sh
    cmake .. -GNinja
    ninja -v
    ```

* If you want to use mingw (GCC) instead of MSVC under windows then [chocolatey](https://chocolatey.org/) can be used to install dependencies for ninja build.

    ```sh
    choco intall -y mingw ninja cmake
    ```
    After that append cmake install path ("C:\Program Files\CMake\bin") to [Environment Variable](https://helpdeskgeek.com/windows-10/add-windows-path-environment-variable/).

