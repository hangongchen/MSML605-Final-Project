#!/usr/bin/env python3
import os
import time
import gc
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

configs = {
    'Small':  dict(embed_dim=128, num_heads=4,  ff_hidden_dim=512),
    'Medium': dict(embed_dim=256, num_heads=8,  ff_hidden_dim=1024),
    'Large':  dict(embed_dim=512, num_heads=8,  ff_hidden_dim=2048),
}
batch_size = 32
seq_length = 128
n_trials   = 50

cpp_cpu_txt = 'Cpp_CPU.txt'
cpp_gpu_txt = 'Cpp_GPU.txt'

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.ln1  = nn.LayerNorm(embed_dim)
        self.ln2  = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        f = self.ff(x)
        return self.ln2(x + f)
def benchmark(device: torch.device):
    process = psutil.Process(os.getpid())
    results = {'Model': [], 'AvgTime(ms)': [], 'PeakMem(MB)': [], 'Throughput(samps/s)': []}

    for name, cfg in configs.items():
        model = TransformerBlock(**cfg).to(device).eval()
        x     = torch.randn(batch_size, seq_length, cfg['embed_dim'], device=device)

        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        else:
            base_rss = process.memory_info().rss
            peak_rss = base_rss

        times = []
        for _ in tqdm(range(n_trials), desc=f"{name} @ {device.type}", leave=False):
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

            if device.type == 'cpu':
                now_rss  = process.memory_info().rss
                peak_rss = max(peak_rss, now_rss)

        avg_t = sum(times) / len(times)
        thrpt = batch_size / (avg_t / 1000)

        if device.type == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        else:
            peak_mem = (peak_rss - base_rss) / (1024**2)

        results['Model'].append(name)
        results['AvgTime(ms)'].append(avg_t)
        results['PeakMem(MB)'].append(peak_mem)
        results['Throughput(samps/s)'].append(thrpt)

        del model, x
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    return pd.DataFrame(results)

print("Running Python CPU benchmark on cpu...")
df_py_cpu = benchmark(torch.device('cpu'))

df_py_gpu = None
if torch.cuda.is_available():
    print("\nRunning Python GPU benchmark on cuda...")
    df_py_gpu = benchmark(torch.device('cuda'))
else:
    print("\nCUDA not available: skipping Python GPU benchmark.")


df_cpp_cpu = pd.read_csv(cpp_cpu_txt) if os.path.exists(cpp_cpu_txt) else None
df_cpp_gpu = pd.read_csv(cpp_gpu_txt) if os.path.exists(cpp_gpu_txt) else None


models = list(configs.keys())
combined_time = pd.DataFrame(index=models)
combined_mem  = pd.DataFrame(index=models)
combined_thr  = pd.DataFrame(index=models)

combined_time['Python_CPU'] = df_py_cpu.set_index('Model')['AvgTime(ms)']
combined_mem ['Python_CPU'] = df_py_cpu.set_index('Model')['PeakMem(MB)']
combined_thr ['Python_CPU'] = df_py_cpu.set_index('Model')['Throughput(samps/s)']

if df_py_gpu is not None:
    combined_time['Python_GPU'] = df_py_gpu.set_index('Model')['AvgTime(ms)']
    combined_mem ['Python_GPU'] = df_py_gpu.set_index('Model')['PeakMem(MB)']
    combined_thr ['Python_GPU'] = df_py_gpu.set_index('Model')['Throughput(samps/s)']

if df_cpp_cpu is not None:
    combined_time['Cpp_CPU'] = df_cpp_cpu.set_index('Model')['AvgTime(ms)']
    combined_mem ['Cpp_CPU'] = df_cpp_cpu.set_index('Model')['PeakMem(MB)']
    combined_thr ['Cpp_CPU'] = df_cpp_cpu.set_index('Model')['Throughput(samps/s)']

if df_cpp_gpu is not None:
    combined_time['Cpp_GPU'] = df_cpp_gpu.set_index('Model')['AvgTime(ms)']
    combined_mem ['Cpp_GPU'] = df_cpp_gpu.set_index('Model')['PeakMem(MB)']
    combined_thr ['Cpp_GPU'] = df_cpp_gpu.set_index('Model')['Throughput(samps/s)']


print("\n-- Combined time table --")
print(combined_time)

print("\n-- Combined memory table --")
print(combined_mem)

print("\n-- Combined throughput table --")
print(combined_thr)


os.makedirs('charts', exist_ok=True)

fig, ax = plt.subplots(figsize=(8,5))
combined_time.plot.bar(ax=ax)
ax.set_title("Execution Time Comparison")
ax.set_ylabel("Time (ms)")
ax.set_xticklabels(models, rotation=0)
fig.tight_layout()
fig.savefig("charts/execution_time.png")

fig, ax = plt.subplots(figsize=(8,5))
combined_mem.plot.bar(ax=ax)
ax.set_title("Peak Memory Footprint (MB)")
ax.set_ylabel("Memory (MB)")
ax.set_xticklabels(models, rotation=0)
fig.tight_layout()
fig.savefig("charts/memory_comparison.png")

fig, ax = plt.subplots(figsize=(8,5))
combined_thr.plot.bar(ax=ax)
ax.set_title("Throughput Comparison")
ax.set_ylabel("Samples/sec")
ax.set_xticklabels(models, rotation=0)
fig.tight_layout()
fig.savefig("charts/throughput_comparison.png")

print("\nSaved comparison charts under ./charts/")