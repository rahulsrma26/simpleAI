#include "snn/snn.hpp"
#include "snn/dataset/mnist.hpp"
#include "snn/util/image.hpp"
#include "snn/util/argparse.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace snn;

    argparse args("autoencoder_mnist",
                  "This is an autoencoder model training program for MNIST dataset.");
    args.add({"mnist_dir"}, "description=directory in which all 4 MNIST files reside");
    args.add({"output_dir"}, "description=directory for output files");
    args.add({"-e", "--epochs"}, "type=integer, default=20, description=number of epoches");
    args.add({"-l", "--learning_rate"},
             "type=real, default=0.001, description=learning rate of the model");
    args.add({"-m", "--load_model"},
             "default='', description=if specified then load the model from the path");
    args.parse(argc, argv);

    auto [trainX, trainY] = dataset::mnist::load(args["mnist_dir"] + "/train");
    trainX.reshape({0, 784});
    cout << "Dataset loaded." << endl;

    models::autoencoder m;
    if (auto model_path = args["load_model"]; model_path != "''") {
        ifstream fin(model_path, ios::binary | ios::in);
        m.load(fin);
        fin.close();
        cout << "Pre-trained model loaded from disk." << endl;
    } else {
        seed(12345);
        m.encoder.add("dense(units=256, input=(784))");
        m.encoder.add("dense(units=32)");
        m.decoder.add("dense(units=256, input=(32))");
        m.decoder.add("dense(units=784)");
        m.compile("loss=cross_entropy(), optimizer=adam(learning_rate=" + args["learning_rate"] +
                  ")");
    }
    m.summary();

    const int epochs = stoi(args["epochs"]);
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainX, "batch_size=32");
    }

    auto out_dir = args["output_dir"] + "/";
    ofstream fout(out_dir + "model.bin", ios::binary | ios::out);
    m.save(fout);
    fout.close();

    const auto examples = trainX.get_shape()[0];
    for (int i = 0; i < 10; i++) {
        auto rand_idx = rand() % examples;
        auto a = math::batch_select(trainX, 1, rand_idx);
        auto b = m.encode(a);
        auto c = m.decode(b);
        a.reshape({28, 28});
        b.reshape({8, 4});
        c.reshape({28, 28});
        save_pgm(out_dir + std::to_string(rand_idx) + "_a.pgm", a);
        save_pgm(out_dir + std::to_string(rand_idx) + "_b.pgm", b);
        save_pgm(out_dir + std::to_string(rand_idx) + "_c.pgm", c);
    }
}
