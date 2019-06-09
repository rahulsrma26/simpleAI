#include "snn/math/tensor.hpp"
#include <iostream>
#include <fstream>

void print_first(const snn::tensor<float>& x) { std::cout << x[{0}] << std::endl; }

int main() {
    using namespace std;
    using namespace snn;

    shape d{2, 3};
    tensor<float> y(d);
    tensor<float> z({3, 4});
    tensor<float> a(z);
    tensor<float> b(std::move(y));
    a = b;
    b = std::move(z);
    cout << y << z << endl;
    b.fill(3);
    cout << b << endl;
    b[{1, 2}] = 5;
    cout << b << endl;
    print_first(b);
    tensor<float> k({4});
    tensor<float> l({200, 300, 400});
    cout << '|' << k << '|' << endl;
    cout << l << endl;
    cout << "l.get_shape() = " << l.get_shape() << endl;
    b.fill(k);
    cout << b << endl;
    int i = 0;
    b.initialize([&]() { return (i++) % 10; });
    cout << b << endl;
    /*sort(b.begin(), b.end());
    cout << b << endl;
    sort(b.rbegin(), b.rend());
    cout << b << endl;*/
    b.reshape({4, 3});
    cout << b << endl;
    b.reshape({2, 2, 3});
    cout << b << endl;
    b.reshape({0, 6});
    cout << b << endl;
    b.reshape({1, 2, 1, 1, 2, 3});
    b.expand_dims(6);
    cout << b.get_shape() << '\n';
    b.squeeze({2, 3});
    cout << b.get_shape() << '\n';
    b.squeeze();
    cout << b.get_shape() << '\n';
    i = 0;
    b.apply([&](auto x) { return x + ((i++) % 10); });
    cout << b << '\n';
    b.transpose({0, 2, 1}); // a c b
    cout << b << '\n';
    b.transpose({1, 0, 2}); // c a b
    cout << b << '\n';
    b.transpose({2, 1, 0}); // b a c
    cout << b << '\n';
    b.transpose({1, 0, 2}); // a b c
    cout << b << '\n';
    vector<int> ntf = {1 * 1 * 1, 1 * 2 * 1, 1 * 3 * 1, 1 * 1 * 2, 1 * 2 * 2, 1 * 3 * 2,
                       2 * 1 * 1, 2 * 2 * 1, 2 * 3 * 1, 2 * 1 * 2, 2 * 2 * 2, 2 * 3 * 2};
    tensor<int> e({2, 3, 2}, std::move(ntf));
    cout << e << '\n';
    e.transpose({1, 0, 2});
    cout << e << '\n';
    tensor<int> f({2, 3}, {0, 1, 2, 3, 4, 5});
    tensor<int> g({3}, {1, 2, 1});
    f += 1;
    cout << f << '\n';
    f += g;
    cout << f << '\n';
    f *= 2;
    cout << f << '\n';
    f *= g;
    cout << f << '\n';
    f -= 1;
    cout << f << '\n';
    f -= g;
    cout << f << '\n';
    f /= 2;
    cout << f << '\n';
    f /= g;
    cout << f << '\n';
    f = 2 + f + g + 3;
    cout << f << '\n';
    f = 3 * f * g * 2;
    cout << f << '\n';
    f = 99 - f - g - 1;
    cout << f << '\n';
    auto f1 = f.cast<float>();
    f1 = 1.0f / (f1 / g.cast<float>()) / 2.0f;
    cout << f1 << '\n';

    ofstream fout("tensors.bin", ios::binary | ios::out);
    e.to_stream(fout);
    g.to_stream(fout);
    fout.close();
    cout << "saved: " << e << '\n';
    cout << "saved: " << g << '\n';

    ifstream fin("tensors.bin", ios::binary | ios::in);
    e = tensor<int>::from_stream(fin);
    g = tensor<int>::from_stream(fin);
    fin.close();
    cout << "read: " << e << '\n';
    cout << "read: " << g << '\n';
}
