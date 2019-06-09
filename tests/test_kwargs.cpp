#include "snn/misc/kwargs.hpp"
#include <iostream>

int main() {
    using namespace std;
    using namespace snn;

    kwargs_type kt({{"mean", "1.0"}});
    cout << kt.size() << '\n';

    kwargs k("mean=1.0, stddev=0.1, xyz  =  -234  ,   ints=(1,2,3), doubles=(1e-5, 0.5e2, -12345.67890, 0.007)  undef_ctr='w,ow!', now='u'");
    cout << k << '\n';
    for (auto val : k.get_int_vector("ints"))
        cout << val << ',';
    cout << '\n';
    for (auto val : k.get_double_vector("doubles"))
        cout << val << ',';
    cout << '\n';

	k = kwargs("xavier(a=500, abc_def='err', k099=-2.1143)");
    cout << k << '\n';

	k = kwargs("init=xavier(a=500, abc_def='err', k099=-2.1143), layer=dense()");
    cout << k << '\n';
    
	auto [f0, p0] = k.get_function("init");
    cout << '"' << f0 << '"' << '=' << '"' << p0 << '"' << '\n';
    auto [f1, p1] = k.get_function("layer");
    cout << '"' << f1 << '"' << '=' << '"' << p1 << '"' << '\n';

    k = kwargs("units=500, use_bias=true, kernel_initializer=xavier(), bias_initializer=zeros(), input=784");
    cout << k << '\n';
}
