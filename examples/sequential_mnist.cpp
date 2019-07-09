#include "snn/snn.hpp"
#include "snn/dataset/mnist.hpp"
#include "snn/util/argparse.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace snn;

    argparse args("sequential_mnist",
                  "This is a sequential model training program for MNIST dataset.");
    args.add({"mnist_dir"}, "description=directory in which all 4 MNIST files reside");
    args.add({"-e", "--epochs"}, "type=integer, default=20, description=number of epoches");
    args.add({"-l", "--learning_rate"},
             "type=real, default=0.001, description=learning rate of the model");
    args.add({"-m", "--load_model"},
             "default='', description=if specified then load the model from the path");
    args.add({"-o", "--output_model"}, "default=model.bin, description=export model path");
    args.parse(argc, argv);

    auto [trainX, trainY] = dataset::mnist::load(args["mnist_dir"] + "/train");
    auto [testX, testY] = dataset::mnist::load(args["mnist_dir"] + "/t10k");
    cout << "Dataset loaded." << endl;

    models::sequential m;
    if (auto model_path = args["load_model"]; model_path != "''") {
        ifstream fin(model_path, ios::binary | ios::in);
        m.load(fin);
        fin.close();
        cout << "Pre-trained model loaded from disk." << endl;
    } else {
        seed(12345);
        m.add("flatten(input=(28,28))");
        m.add("dense(units=400, activation=relu())");
        m.add("dropout(rate=0.3)");
        m.add("dense(units=50, activation=relu())");
        m.add("dropout(rate=0.5)");
        m.add("dense(units=10)");
        m.compile("loss=cross_entropy(), optimizer=adam(learning_rate=" + args["learning_rate"] +
                  ")");
    }
    m.summary();

    const int epochs = stoi(args["epochs"]);
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainY, "batch_size=32");
        m.run(testX, testY, "batch_size=128, train=false");
    }

    ofstream fout(args["output_model"], ios::binary | ios::out);
    m.save(fout);
    fout.close();
}
