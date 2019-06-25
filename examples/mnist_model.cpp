#include "snn/model/sequential.hpp"
#include "snn/dataset/mnist.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace snn;

    if (argc < 2) {
        cout << "Usages:" << '\n';
        cout << argv[0] << " <mnist-dir>" << '\n';
        return 0;
    }

    auto data_dir = string(argv[1]);
    auto [trainX, trainY] = dataset::mnist::load(data_dir + "/train");
    auto [testX, testY] = dataset::mnist::load(data_dir + "/t10k");
    cout << "Dataset loaded." << endl;

    models::sequential m;
    ifstream fin("model.bin", ios::binary | ios::in);
    if (fin.good()) {
        m.load(fin);
        fin.close();
        cout << "Pre-trained model loaded from disk." << endl;
    } else {
        seed(12345);
        string learning_rate = argc >= 4 ? argv[3] : "0.001";
        string loss = argc >= 5 ? argv[4] : "cross_entropy";
        m.add("flatten(input=(28,28))");
        m.add("dense(units=400)");
        m.add("dropout(rate=0.3, activation=relu())");
        m.add("dense(units=50)");
        m.add("dropout(rate=0.5, activation=relu())");
        m.add("dense(units=10)");
        m.compile("loss=" + loss + "(), optimizer=adam(learning_rate=" + learning_rate + ")");
        m.summary();
    }

    const int epochs = argc >= 3 ? stoi(argv[2]) : 20;
    for (int epoch = 1; epoch <= epochs; epoch++) {
        cout << "Epoch: " << epoch << '/' << epochs << endl;
        m.run(trainX, trainY, "batch_size=32");
        m.run(testX, testY, "batch_size=128, train=false");
    }

    ofstream fout("model.bin", ios::binary | ios::out);
    m.save(fout);
    fout.close();
}
