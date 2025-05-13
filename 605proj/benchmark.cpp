//Results will be saved to Cpp_CPU.txt and Cpp_GPU.txt
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cmath>
#include <string>
#include <sys/resource.h>
#include <numeric>

using namespace Eigen;
using namespace std;
class TransformerBlock {
public:
    TransformerBlock(int embed_dim, int ff_hidden_dim) {
        Wq   = MatrixXd::Random(embed_dim, embed_dim);
        Wk   = MatrixXd::Random(embed_dim, embed_dim);
        Wv   = MatrixXd::Random(embed_dim, embed_dim);
        Wff1 = MatrixXd::Random(embed_dim, ff_hidden_dim);
        Wff2 = MatrixXd::Random(ff_hidden_dim, embed_dim);
    }
    MatrixXd forward(const MatrixXd& x) {
        MatrixXd Q = x * Wq;
        MatrixXd K = x * Wk;
        MatrixXd V = x * Wv;
        MatrixXd scores = (Q * K.transpose()) / sqrt(double(Q.cols()));
        scores = scores.array().exp();
        scores = scores.array().colwise() / scores.rowwise().sum().array();
        MatrixXd context = scores * V;
        MatrixXd ff = (context * Wff1).unaryExpr([](double v){ return v > 0 ? v : 0; });
        ff = ff * Wff2;
        return ff;
    }
private:
    MatrixXd Wq, Wk, Wv, Wff1, Wff2;
};
static long getCurrentRSS_kB() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}
struct Config {
    string name;
    int embed_dim;
    int ff_hidden_dim;
};
int main() {
    const int batch_size = 32;
    const int seq_length = 128;
    const int runs       = 100;

    vector<Config> configs = {
            {"Small",  128,  512},
            {"Medium", 256, 1024},
            {"Large",  512, 2048}
    };
    ofstream cpu_out("Cpp_CPU.txt", ios::out);
    cpu_out << "Model,AvgTime_ms,PeakMemory_MB,Throughput_samples_per_sec\n";
    for (auto& cfg : configs) {
        TransformerBlock model(cfg.embed_dim, cfg.ff_hidden_dim);
        MatrixXd input = MatrixXd::Random(batch_size * seq_length, cfg.embed_dim);
        for (int i = 0; i < 10; ++i)
            model.forward(input);
        vector<double> times;
        times.reserve(runs);
        long peak_rss_kB = 0;
        for (int i = 0; i < runs; ++i) {
            auto t0 = chrono::high_resolution_clock::now();
            model.forward(input);
            auto t1 = chrono::high_resolution_clock::now();
            double ms = chrono::duration<double, milli>(t1 - t0).count();
            times.push_back(ms);
            long rss = getCurrentRSS_kB();
            if (rss > peak_rss_kB) peak_rss_kB = rss;
        }
        double avg_ms      = accumulate(times.begin(), times.end(), 0.0) / times.size();
        double peak_mem_MB = peak_rss_kB / 1024.0;
        double throughput  = batch_size / (avg_ms / 1000.0);
        cout << "=== Benchmark: " << cfg.name << " model ===\n"
             << "Average execution time: " << avg_ms      << " ms\n"
             << "Peak RSS (CPU mem):     " << peak_mem_MB << " MB\n"
             << "Throughput:             " << throughput  << " samples/sec\n\n";
        cpu_out << cfg.name << ","
                << avg_ms      << ","
                << peak_mem_MB << ","
                << throughput  << "\n";
    }
    cpu_out.close();
    return 0;
}