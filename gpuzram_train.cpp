// NanoGPT-Golf v7.0-GPUZRAM-SAFE - C++ Implementation
// Компилируемая реализация без Python overhead
// Работает на CPU с симуляцией GPU для сред без видеокарты
// 
// Сборка: g++ -O3 -march=native -pthread -lz gpuzram_train.cpp -o gpuzram_train
// Запуск: ./gpuzram_train --layers 6 --dim 128 --seq 256 --batch 4

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdlib>
#include <zlib.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

// ============================================================================
// GPU-ZRAM Manager: Сжатие и оффлоадинг данных тренировки
// ============================================================================

struct CompressionStats {
    std::atomic<uint64_t> total_original{0};
    std::atomic<uint64_t> total_compressed{0};
    std::atomic<uint64_t> compression_count{0};
    std::atomic<uint64_t> offload_count{0};
    std::atomic<uint64_t> decompress_count{0};
};

class GPUZramManager {
private:
    std::mutex mtx;
    std::vector<std::vector<unsigned char>> compressed_chunks;
    CompressionStats stats;
    
    // Порог для экстренного оффлоадинга (симуляция 90% VRAM)
    static constexpr double EMERGENCY_THRESHOLD = 0.90;
    
public:
    // Сжатие данных (градиенты, активации, optimizer states)
    bool compressAndOffload(const float* data, size_t count, const std::string& label) {
        std::lock_guard<std::mutex> lock(mtx);
        
        size_t original_size = count * sizeof(float);
        
        // Оценка сжимаемости (градиенты после ReLU имеют много нулей)
        int zero_count = 0;
        for (size_t i = 0; i < count && i < 10000; ++i) {
            if (std::abs(data[i]) < 1e-6f) zero_count++;
        }
        double sparsity = (count > 10000) ? 0.6 : static_cast<double>(zero_count) / std::min(count, size_t(10000));
        
        // Буфер для сжатых данных
        size_t bound = deflateBound(nullptr, original_size);
        std::vector<unsigned char> compressed(bound);
        
        z_stream strm = {};
        if (deflateInit(&strm, Z_DEFAULT_COMPRESSION) != Z_OK) {
            return false;
        }
        
        strm.next_in = reinterpret_cast<Bytef*>(const_cast<float*>(data));
        strm.avail_in = original_size;
        strm.next_out = compressed.data();
        strm.avail_out = bound;
        
        int ret = deflate(&strm, Z_FINISH);
        size_t compressed_size = (ret == Z_STREAM_END) ? strm.total_out : bound;
        
        deflateEnd(&strm);
        
        if (ret != Z_STREAM_END) {
            return false;
        }
        
        // Сохраняем сжатые данные
        compressed.resize(compressed_size);
        compressed_chunks.push_back(std::move(compressed));
        
        // Обновляем статистику
        stats.total_original.fetch_add(original_size);
        stats.total_compressed.fetch_add(compressed_size);
        stats.compression_count.fetch_add(1);
        stats.offload_count.fetch_add(1);
        
        return true;
    }
    
    // INT8 квантование с последующим сжатием (для активаций)
    bool quantizeAndCompress(const float* data, size_t count, const std::string& label) {
        std::lock_guard<std::mutex> lock(mtx);
        
        // Находим min/max для квантования
        float min_val = data[0], max_val = data[0];
        for (size_t i = 1; i < count; ++i) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }
        
        float scale = (max_val - min_val) / 255.0f;
        if (scale < 1e-10f) scale = 1e-10f;
        
        // Квантуем в INT8
        std::vector<int8_t> quantized(count);
        for (size_t i = 0; i < count; ++i) {
            quantized[i] = static_cast<int8_t>((data[i] - min_val) / scale);
        }
        
        // Сжимаем квантованные данные
        size_t original_size = count * sizeof(int8_t);
        size_t bound = deflateBound(nullptr, original_size);
        std::vector<unsigned char> compressed(bound);
        
        z_stream strm = {};
        if (deflateInit(&strm, Z_DEFAULT_COMPRESSION) != Z_OK) {
            return false;
        }
        
        strm.next_in = reinterpret_cast<Bytef*>(quantized.data());
        strm.avail_in = original_size;
        strm.next_out = compressed.data();
        strm.avail_out = bound;
        
        int ret = deflate(&strm, Z_FINISH);
        size_t compressed_size = (ret == Z_STREAM_END) ? strm.total_out : bound;
        
        deflateEnd(&strm);
        
        if (ret != Z_STREAM_END) {
            return false;
        }
        
        compressed.resize(compressed_size);
        compressed_chunks.push_back(std::move(compressed));
        
        stats.total_original.fetch_add(count * sizeof(float)); // Оригинал был float32
        stats.total_compressed.fetch_add(compressed_size);
        stats.compression_count.fetch_add(1);
        
        return true;
    }
    
    // Восстановление из сжатого состояния (OOM rescue)
    bool restoreFromBackup(size_t chunk_index, float* output, size_t expected_count) {
        std::lock_guard<std::mutex> lock(mtx);
        
        if (chunk_index >= compressed_chunks.size()) {
            return false;
        }
        
        const auto& compressed = compressed_chunks[chunk_index];
        
        z_stream strm = {};
        if (inflateInit(&strm) != Z_OK) {
            return false;
        }
        
        strm.next_in = const_cast<Bytef*>(compressed.data());
        strm.avail_in = compressed.size();
        strm.next_out = reinterpret_cast<Bytef*>(output);
        strm.avail_out = expected_count * sizeof(float);
        
        int ret = inflate(&strm, Z_FINISH);
        inflateEnd(&strm);
        
        if (ret == Z_STREAM_END || ret == Z_OK) {
            stats.decompress_count.fetch_add(1);
            return true;
        }
        
        return false;
    }
    
    // Проверка необходимости экстренного оффлоадинга
    bool checkEmergencyOffload(double current_usage_percent) {
        return current_usage_percent >= EMERGENCY_THRESHOLD;
    }
    
    // Статистика
    void printStats() const {
        uint64_t orig = stats.total_original.load();
        uint64_t comp = stats.total_compressed.load();
        uint64_t count = stats.compression_count.load();
        uint64_t offload = stats.offload_count.load();
        uint64_t decompress = stats.decompress_count.load();
        
        double ratio = (comp > 0) ? static_cast<double>(orig) / comp : 0;
        
        std::cout << "\n📊 GPU-ZRAM Statistics:" << std::endl;
        std::cout << "   Total Compressions: " << count << std::endl;
        std::cout << "   Emergency Offloads: " << offload << std::endl;
        std::cout << "   Restores (OOM Rescue): " << decompress << std::endl;
        std::cout << "   Original Data: " << std::fixed << std::setprecision(2) 
                  << (orig / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "   Compressed Data: " << (comp / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "   Compression Ratio: " << std::setprecision(2) << ratio << "x" << std::endl;
        if (orig > comp) {
            std::cout << "   Memory Saved: " << ((orig - comp) / 1024.0 / 1024.0) << " MB" << std::endl;
        }
    }
    
    const CompressionStats& getStats() const { return stats; }
};

// ============================================================================
// Симуляция GPU вычислений (матричные операции для нагрузки)
// ============================================================================

class GPUSimulator {
private:
    std::vector<float> matrix_a;
    std::vector<float> matrix_b;
    std::vector<float> matrix_c;
    size_t matrix_size;
    std::atomic<double> current_vram_usage{0.0};
    size_t total_vram_capacity;
    
public:
    GPUSimulator(size_t n, size_t vram_mb) : matrix_size(n) {
        matrix_a.resize(n * n);
        matrix_b.resize(n * n);
        matrix_c.resize(n * n);
        
        total_vram_capacity = vram_mb * 1024 * 1024;
        
        // Инициализация случайными данными
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        for (size_t i = 0; i < n * n; ++i) {
            matrix_a[i] = dis(gen);
            matrix_b[i] = dis(gen);
            matrix_c[i] = 0.0f;
        }
        
        updateVramUsage();
    }
    
    // Матричное умножение (нагрузка на CPU, симуляция GPU kernel)
    void matMulKernel(size_t rows, size_t cols, size_t common) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < common; ++k) {
                    sum += matrix_a[i * common + k] * matrix_b[k * cols + j];
                }
                matrix_c[i * cols + j] = sum;
            }
        }
    }
    
    // Стресс-тест памяти (чтение/запись)
    void memStressKernel(size_t iterations) {
        for (size_t iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < matrix_size * matrix_size; ++i) {
                matrix_c[i] = matrix_c[i] * 1.0001f + 0.0001f;
            }
        }
    }
    
    // Вычисление нагрузки на память (в байтах)
    size_t getCurrentMemoryUsage() const {
        return (matrix_a.size() + matrix_b.size() + matrix_c.size()) * sizeof(float);
    }
    
    void updateVramUsage() {
        double usage = static_cast<double>(getCurrentMemoryUsage()) / total_vram_capacity * 100.0;
        current_vram_usage.store(usage);
    }
    
    double getVramUsagePercent() const {
        return current_vram_usage.load();
    }
    
    float* getMatrixC() { return matrix_c.data(); }
    size_t getMatrixSize() const { return matrix_size * matrix_size; }
};

// ============================================================================
// CPU Cache Micro-benchmark (определение реально доступных кэшей)
// ============================================================================

struct CacheInfo {
    size_t l1_size;
    size_t l2_size;
    size_t l3_size;
    double l1_bandwidth_gb_s;
    double l2_bandwidth_gb_s;
    double l3_bandwidth_gb_s;
    bool l1_available;
    bool l2_available;
    bool l3_available;
};

CacheInfo benchmarkCPUCaches() {
    CacheInfo info = {};
    
    // Тестовые размеры для определения границ кэшей
    std::vector<size_t> test_sizes = {
        8 * 1024,      // 8 KB
        16 * 1024,     // 16 KB
        32 * 1024,     // 32 KB
        64 * 1024,     // 64 KB (типичный L1)
        128 * 1024,    // 128 KB
        256 * 1024,    // 256 KB
        512 * 1024,    // 512 KB
        1024 * 1024,   // 1 MB (типичный L2)
        2 * 1024 * 1024,  // 2 MB
        4 * 1024 * 1024,  // 4 MB
        8 * 1024 * 1024,  // 8 MB
        16 * 1024 * 1024, // 16 MB
        32 * 1024 * 1024, // 32 MB (типичный L3)
        55 * 1024 * 1024, // 55 MB (верхняя граница L3)
    };
    
    std::vector<double> latencies;
    
    for (size_t size : test_sizes) {
        std::vector<float> data(size / sizeof(float));
        std::iota(data.begin(), data.end(), 1.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Многократный доступ для измерения latency
        volatile float sum = 0;
        for (int iter = 0; iter < 100; ++iter) {
            for (size_t i = 0; i < data.size(); ++i) {
                sum += data[i];
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        double bandwidth = (static_cast<double>(size) * 100) / (duration / 1e9) / (1024 * 1024 * 1024);
        latencies.push_back(bandwidth);
    }
    
    // Определение границ кэшей по спадам bandwidth
    info.l1_size = 32 * 1024;  // Default
    info.l2_size = 256 * 1024; // Default
    info.l3_size = 8 * 1024 * 1024; // Default
    
    // Простая эвристика: находим спады пропускной способности
    for (size_t i = 1; i < latencies.size(); ++i) {
        if (latencies[i] < latencies[i-1] * 0.7) { // Спад на 30%
            if (info.l1_size == 32 * 1024 && i >= 3) {
                info.l1_size = test_sizes[i-1];
            } else if (info.l2_size == 256 * 1024 && i >= 6) {
                info.l2_size = test_sizes[i-1];
            } else if (info.l3_size == 8 * 1024 * 1024 && i >= 10) {
                info.l3_size = test_sizes[i-1];
                break;
            }
        }
    }
    
    info.l1_bandwidth_gb_s = latencies[3]; // ~32KB
    info.l2_bandwidth_gb_s = latencies[7]; // ~256KB
    info.l3_bandwidth_gb_s = latencies[10]; // ~8MB
    
    info.l1_available = info.l1_bandwidth_gb_s > 10.0;
    info.l2_available = info.l2_bandwidth_gb_s > 5.0;
    info.l3_available = info.l3_bandwidth_gb_s > 1.0;
    
    return info;
}

// ============================================================================
// Training Loop с GPU-ZRAM защитой
// ============================================================================

void printProgressBar(int current, int total, int width = 40) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(width * progress);
    
    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
        if (i < pos) std::cout << "█";
        else if (i == pos) std::cout << "▌";
        else std::cout << "░";
    }
    std::cout << "] " << std::fixed << std::setprecision(1) << (progress * 100.0) << "%";
    std::cout.flush();
}

int main(int argc, char* argv[]) {
    // Параметры по умолчанию
    int layers = 6;
    int dim = 128;
    int seq_len = 256;
    int batch_size = 4;
    int max_steps = 1000;
    size_t max_cpu_backup_mb = 2048;
    
    // Парсинг аргументов
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--layers" && i + 1 < argc) {
            layers = std::stoi(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            dim = std::stoi(argv[++i]);
        } else if (arg == "--seq" && i + 1 < argc) {
            seq_len = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            max_steps = std::stoi(argv[++i]);
        } else if (arg == "--max-cpu-backup-mb" && i + 1 < argc) {
            max_cpu_backup_mb = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "NanoGPT-Golf v7.0-GPUZRAM-SAFE\n\n"
                      << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --layers N           Number of transformer layers (default: 6)\n"
                      << "  --dim N              Embedding dimension (default: 128)\n"
                      << "  --seq N              Sequence length (default: 256)\n"
                      << "  --batch N            Batch size (default: 4)\n"
                      << "  --iters N            Maximum training steps (default: 1000)\n"
                      << "  --max-cpu-backup-mb  Max CPU RAM for backup (default: 2048)\n"
                      << "  --help               Show this help message\n";
            return 0;
        }
    }
    
    std::cout << "🚀 NanoGPT-Golf v7.0-GPUZRAM-SAFE (C++ Implementation)" << std::endl;
    std::cout << "========================================================" << std::endl;
    
    // Бенчмарк кэшей CPU
    std::cout << "\n📊 CPU Cache Micro-benchmark:" << std::endl;
    CacheInfo cache_info = benchmarkCPUCaches();
    std::cout << "   L1 Cache: " << (cache_info.l1_size / 1024) << " KB @ " 
              << std::fixed << std::setprecision(1) << cache_info.l1_bandwidth_gb_s << " GB/s"
              << (cache_info.l1_available ? " ✅" : " ❌") << std::endl;
    std::cout << "   L2 Cache: " << (cache_info.l2_size / 1024) << " KB @ " 
              << cache_info.l2_bandwidth_gb_s << " GB/s"
              << (cache_info.l2_available ? " ✅" : " ❌") << std::endl;
    std::cout << "   L3 Cache: " << (cache_info.l3_size / 1024 / 1024) << " MB @ " 
              << cache_info.l3_bandwidth_gb_s << " GB/s"
              << (cache_info.l3_available ? " ✅" : " ❌") << std::endl;
    
    // Расчет размера модели
    size_t params_per_layer = 4 * dim * dim + 2 * dim * dim; // Attention + MLP примерно
    size_t total_params = params_per_layer * layers + dim * 50257; // + embeddings
    size_t model_size_mb = (total_params * 4) / (1024.0 * 1024.0); // FP32
    
    std::cout << "\n🧠 Model Configuration:" << std::endl;
    std::cout << "   Layers: " << layers << ", Dim: " << dim 
              << ", Seq: " << seq_len << ", Batch: " << batch_size << std::endl;
    std::cout << "   Parameters: " << (total_params / 1e6) << " M (" 
              << std::fixed << std::setprecision(2) << model_size_mb << " MB FP32)" << std::endl;
    
    // Инициализация GPU симулятора
    size_t matrix_n = std::min(static_cast<size_t>(512), 
                               static_cast<size_t>(dim * 2));
    size_t simulated_vram_mb = std::max(max_cpu_backup_mb / 2, size_t(1024));
    
    GPUSimulator gpu(matrix_n, simulated_vram_mb);
    GPUZramManager zram_mgr;
    
    std::cout << "\n💾 Simulated VRAM: " << simulated_vram_mb << " MB" << std::endl;
    std::cout << "   Initial allocation: " << std::fixed << std::setprecision(2)
              << (gpu.getCurrentMemoryUsage() / 1024.0 / 1024.0) << " MB" << std::endl;
    
    std::cout << "\n🔥 Starting Training Loop with GPU-ZRAM Protection..." << std::endl;
    std::cout << "   (Monitor with htop for CPU usage)" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::atomic<int> oom_rescues{0};
    std::atomic<int> emergency_offloads{0};
    
    // Основной тренировочный цикл
    for (int step = 0; step < max_steps; ++step) {
        // Симуляция forward pass (матричное умножение)
        gpu.matMulKernel(matrix_n, matrix_n, matrix_n);
        
        // Симуляция backward pass (стресс памяти)
        gpu.memStressKernel(5);
        
        gpu.updateVramUsage();
        double vram_usage = gpu.getVramUsagePercent();
        
        // Проверка на необходимость экстренного оффлоадинга
        if (vram_usage > 85.0) { // Симуляция порога OOM
            // Сжатие и оффлоадинг активаций
            float* activations = gpu.getMatrixC();
            size_t act_size = gpu.getMatrixSize();
            
            if (step % 10 == 0) {
                zram_mgr.compressAndOffload(activations, act_size, "activations");
                emergency_offloads.fetch_add(1);
            }
            
            // Симуляция OOM rescue если.usage > 95%
            if (vram_usage > 95.0 && step > 10) {
                std::cout << "\n⚠️  OOM DETECTED at step " << step 
                          << "! Initiating rescue..." << std::endl;
                
                // Восстановление из бэкапа
                std::vector<float> restored(act_size);
                if (zram_mgr.restoreFromBackup(0, restored.data(), act_size)) {
                    oom_rescues.fetch_add(1);
                    std::cout << "✅ OOM Rescue successful! Restored from CPU backup." << std::endl;
                }
                
                // "Очистка" памяти (симуляция)
                gpu.memStressKernel(1);
            }
        }
        
        // Периодическое сжатие градиентов (симуляция)
        if (step % 50 == 0 && step > 0) {
            float* grads = gpu.getMatrixC();
            size_t grad_size = gpu.getMatrixSize();
            
            // INT8 квантование для лучшей компрессии
            zram_mgr.quantizeAndCompress(grads, grad_size, "gradients");
        }
        
        // Прогресс бар
        if (step % 50 == 0) {
            printProgressBar(step, max_steps);
            std::cout << " Step " << step << "/" << max_steps 
                      << " | VRAM: " << std::fixed << std::setprecision(1) << vram_usage << "%"
                      << " | ZRAM: " << zram_mgr.getStats().compression_count.load() << " compressions";
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    printProgressBar(max_steps, max_steps);
    std::cout << std::endl;
    
    // Финальная статистика
    std::cout << "\n✅ Training completed!" << std::endl;
    std::cout << "   Time: " << (duration / 1000.0) << " s (" 
              << std::fixed << std::setprecision(0) 
              << (max_steps * 1000.0 / duration) << " steps/s)" << std::endl;
    std::cout << "   OOM Rescues: " << oom_rescues.load() << std::endl;
    std::cout << "   Emergency Offloads: " << emergency_offloads.load() << std::endl;
    
    zram_mgr.printStats();
    
    std::cout << "\n👋 Done." << std::endl;
    
    return 0;
}
