#include "snn/snn.hpp"
#include "snn/dataset/mnist.hpp"
#include "snn/util/image.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace snn;

    if (argc < 3) {
        cout << "Usages:" << '\n';
        cout << argv[0] << " <mnist-dir> <output-dir> [epochs=20]" << '\n';
        return 0;
    }

    auto data_dir = string(argv[1]);
    auto [trainX, trainY] = dataset::mnist::load(data_dir + "/train");
    // auto [testX, testY] = dataset::mnist::load(data_dir + "/t10k");
    trainX.reshape({0, 784});
    cout << "Dataset loaded." << endl;

    models::autoencoder m;
    ifstream fin("model.bin", ios::binary | ios::in);
    if (fin.good()) {
        m.load(fin);
        fin.close();
        cout << "Pre-trained model loaded from disk." << endl;
    } else {
        seed(12345);
        m.encoder.add("dense(units=256, input=(784))");
        m.encoder.add("dense(units=32)");
        m.decoder.add("dense(units=256, input=(32))");
        m.decoder.add("dense(units=784)");
        m.compile("loss=cross_entropy(), optimizer=adam(learning_rate=0.001)");
        m.summary();
    }

    const int epochs = argc >= 4 ? stoi(argv[3]) : 20;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainX, "batch_size=32");
    }

    ofstream fout("model.bin", ios::binary | ios::out);
    m.save(fout);
    fout.close();

    auto out_dir = string(argv[2]);
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
