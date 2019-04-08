#include "mnistDataSet.h"
#include "simpleNN.h"
#include <cstdio>
#include <chrono>

int main(int argc, char* argv[]) {
    using namespace std;
    using namespace chrono;
    using namespace simpleNN;

    if (argc != 2) {
        cout << "Usages:" << '\n';
        cout << argv[0] << " <mnist-dir>" << '\n';
        return 0;
    }

    NeuralNetwork nn(784, loss::crossEntropy);
    nn.add(make_unique<DenseLayer>(400));
    nn.add(make_unique<Dropout>(0.3f));
    nn.add(make_unique<DenseLayer>(10));

    auto data_dir = string(argv[1]);
    auto [trainX, trainY] = dataset::MnistDataSet::get(data_dir + "/train-images-idx3-ubyte",
                                                       data_dir + "/train-labels-idx1-ubyte");
    auto [testX, testY] = dataset::MnistDataSet::get(data_dir + "/t10k-images-idx3-ubyte",
                                                     data_dir + "/t10k-labels-idx1-ubyte");

    printf("+-------+---------------------+---------------------+------------+\n");
    printf("|       |        train        |         test        |            |\n");
    printf("| sweep |     loss | accuracy |     loss | accuracy | time (sec) |\n");
    printf("|-------+----------+----------+----------+----------+------------|\n");

    constexpr int SWEEPS = 30;
    constexpr size_t BATCH_SIZE = 10;
    real learninngRate = 0.3f;

    for (int i = 1; i <= SWEEPS; ++i) {
        auto start = high_resolution_clock::now();
        nn.train(trainX, trainY, BATCH_SIZE, learninngRate);
        printf("| %5d |", i);
        auto stats = nn.getStats(trainX, trainY);
        printf(" %8.5f | %0.6f |", stats.loss, stats.accuracy);
        stats = nn.getStats(testX, testY);
        printf(" %8.5f | %0.6f |", stats.loss, stats.accuracy);
        learninngRate *= .97f;
        double time =
            duration_cast<nanoseconds>(high_resolution_clock::now() - start).count() / 1e9;
        printf(" %10.5f |\n", time);
    }

    return 0;
}

/*

Performance
            Quadratic           Hillinger           Cross-entropy       Cross-entropy + Dropout
Epochs	train	    test	train	    test	train	    test	train	    test
     1	0.381	    0.3771	0.387517	0.3818	0.947433	0.938	0.938583	0.9349
     2	0.38725	    0.3825	0.4052	    0.3984	0.9688	    0.9538	0.95605	    0.9489
     3	0.393183	0.3875	0.445467	0.4386	0.974317	0.9558	0.962033	0.9528
     4	0.395467	0.3893	0.4639	    0.4562	0.982417	0.9606	0.967983	0.9597
     5	0.408467	0.4018	0.46995	    0.4623	0.987267	0.9646	0.97145	    0.9621
     6	0.423717	0.4168	0.47305	    0.4651	0.99035	    0.9637	0.974267	0.9638
     7	0.4479	    0.4411	0.474217	0.4668	0.994133	0.9686	0.97705	    0.9638
     8	0.461717	0.4556	0.47515	    0.4678	0.993583	0.9644	0.978883	0.9659
     9	0.4679	    0.4605	0.4766	    0.4693	0.996133	0.9675	0.980767	0.9676
    10	0.4711	    0.4634	0.477617	0.4694	0.997	    0.967	0.982583	0.9688
    11	0.472083	0.4648	0.478283	0.47	0.998017	0.9695	0.984283	0.9686
    12	0.473083	0.4661	0.47915	    0.4705	0.998533	0.9691	0.985117	0.9691
    13	0.474567	0.4667	0.4799	    0.4705	0.998683	0.9699	0.98625	    0.9712
    14	0.475217	0.4678	0.480233	0.4712	0.99895	    0.97	0.987467	0.9704
    15	0.475883	0.4678	0.4807	    0.4715	0.998867	0.9691	0.987317	0.971
    16	0.476067	0.4682	0.480983	0.4712	0.999183	0.9702	0.988617	0.9714
    17	0.476433	0.4686	0.4816	    0.4714	0.999217	0.9693	0.9899	    0.9729
    18	0.4773	    0.4693	0.482283	0.4729	0.999283	0.9705	0.98985	    0.9733
    19	0.477467	0.4693	0.483283	0.4738	0.99935	    0.9699	0.9907	    0.9729
    20	0.477767	0.4691	0.487783	0.4799	0.999367	0.9714	0.99105	    0.9743
    21	0.478017	0.4701	0.587417	0.5794	0.9994	    0.9704	0.9914	    0.9742
    22	0.478267	0.4703	0.5909	    0.5832	0.999417	0.9706	0.991633	0.9741
    23	0.478567	0.4703	0.5914	    0.5838	0.99945	    0.9707	0.992083	0.9743
    24	0.478717	0.4705	0.5917	    0.5839	0.99945	    0.9705	0.992467	0.9751
    25	0.478933	0.4708	0.5919	    0.584	0.999467	0.9706	0.9925	    0.9744
    26	0.478867	0.4711	0.592183	0.5849	0.999483	0.9712	0.993017	0.975
    27	0.479167	0.4714	0.59255	    0.5848	0.999483	0.9711	0.99325	    0.9746
    28	0.47935	    0.4716	0.592833	0.585	0.999483	0.9708	0.993483	0.9744
    29	0.479517	0.4713	0.592883	0.5846	0.999483	0.9708	0.99385	    0.9749
    30	0.479733	0.4716	0.593033	0.5849	0.9995	    0.9711	0.993833	0.9758

*/