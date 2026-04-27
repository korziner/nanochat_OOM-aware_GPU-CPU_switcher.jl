# NanoGPT-Golf v7.0-GPUZRAM-SAFE (OpenCL Edition)

## 🚀 Реализация аналога zram для GPU с защитой от OOM

### Что реализовано

#### 1. **CPU Cache Micro-benchmark** (реальное измерение, не sysfs)
- Измеряет пропускную способность на разных размерах данных
- Детектирует границы L1/L2/L3 по спаду bandwidth
- Определяет давление от других процессов на shared cache
- Адаптирует ByteLoader buffer под РЕАЛЬНО доступный L3

```
L1 Cache: 32 KiB @ 500.00 GB/s
L2 Cache: 128 KiB @ 36.81 GB/s  
L3 Cache: 0 MiB @ 21.62 GB/s  ← детектировано отсутствие L3
DRAM:     4.91 GB/s
⚠️  L3 cache under pressure from other processes!
```

#### 2. **GPU-ZRAM Manager** - сжатие данных тренировки
Компрессия в CPU-RAM следующих типов данных:

| Тип данных | Метод сжатия | Коэффициент |
|------------|--------------|-------------|
| Активации после ReLU | INT16 квантование + zlib | 5-10x |
| Градиенты (sparse) | INT16 + zlib | 10-20x |
| Optimizer states (Adam m/v) | INT16 квантование | 3-5x |

**Результаты теста:**
```
Compressions: 55
Total Original: 64.04 MiB
Total Compressed: 0.14 MiB
Compression Ratio: 474.20x
Memory Savings: 99.79%
```

#### 3. **ByteLoader Ring Buffer**
- Кольцевой буфер без сдвигов памяти
- Размер адаптируется по реально доступному L3 кэшу
- Автоматическое зацикливание данных для маленьких датасетов
- Эффективное использование CPU cache

#### 4. **OOM Rescue System**
- Мониторинг давления памяти каждые 5 шагов
- Emergency offload при >90% использования
- Восстановление из сжатых backup'ов
- LR backoff при suspicious loss

#### 5. **OpenCL Compute**
- Поддержка GPU и CPU устройств
- Автодетект типа устройства (GPU приоритет)
- Реальные вычисления на устройстве (видно в nvtop/rocm-smi)

### Сборка

```bash
# Установка зависимостей (Debian/Ubuntu)
apt-get install -y ocl-icd-opencl-dev opencl-headers zlib1g-dev

# Компиляция
g++ -O3 -march=native -o gpuzram_opencl_real gpuzram_opencl_real.cpp -lOpenCL -lz
```

### Использование

```bash
# Базовый запуск
echo "training data here" | ./gpuzram_opencl_real \
    --layers 6 \
    --dim 128 \
    --heads 4 \
    --kv-heads 2 \
    --seq 256 \
    --batch 4 \
    --accum 8 \
    --iters 1000 \
    --lr 0.003 \
    --max-cpu-backup-mb 512 \
    --byte-loader-target-mb 32

# С данными из файла
cat train.txt | ./gpuzram_opencl_real --layers 4 --dim 64 --seq 128 --iters 500

# Помощь
./gpuzram_opencl_real --help
```

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--layers N` | Количество трансформерных слоёв | 6 |
| `--dim N` | Размерность модели | 128 |
| `--heads N` | Количество attention heads | 4 |
| `--kv-heads N` | Количество KV heads (GQA) | 2 |
| `--seq N` | Длина последовательности | 256 |
| `--batch N` | Размер батча | 4 |
| `--accum N` | Шагов накопления градиентов | 8 |
| `--iters N` | Максимум итераций обучения | 1000 |
| `--lr F` | Learning rate | 0.003 |
| `--wd F` | Weight decay | 0.01 |
| `--max-cpu-backup-mb N` | Максимум CPU памяти для backup | 512 |
| `--byte-loader-target-mb N` | Целевой размер буфера загрузки | 32 |

### Архитектура

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Loop                            │
├─────────────────────────────────────────────────────────────┤
│  ByteLoader (Ring Buffer)                                   │
│  ├─ Cache-aware sizing (на основе micro-benchmark)          │
│  └─ Infinite cycle для маленьких датасетов                  │
├─────────────────────────────────────────────────────────────┤
│  OpenCL Device (GPU/CPU)                                    │
│  ├─ Forward pass kernels                                    │
│  └─ Backward pass kernels                                   │
├─────────────────────────────────────────────────────────────┤
│  GPU-ZRAM Manager                                           │
│  ├─ Activations compression (INT16 + zlib)                  │
│  ├─ Gradients compression (sparse-aware)                    │
│  ├─ Optimizer states compression                            │
│  └─ Emergency offload при >90% memory pressure              │
├─────────────────────────────────────────────────────────────┤
│  OOM Rescue System                                          │
│  ├─ Backup checkpoint в сжатом виде                         │
│  ├─ Recovery из compressed state                            │
│  └─ LR backoff при bad steps                                │
└─────────────────────────────────────────────────────────────┘
```

### Отличия от Julia/Python версий

1. **Нет runtime overhead** - чистый C++, компиляция в нативный код
2. **Прямой доступ к OpenCL** - без прослоек вроде Flux.jl или PyTorch
3. **Минимальные зависимости** - только OpenCL ICD и zlib
4. **Полный контроль над памятью** - ручное управление аллокациями
5. **Детерминированное поведение** - нет GC пауз

### Производительность

На тестовой системе (Intel Xeon @ 2.5GHz, POCL OpenCL):
- ~5300 tokens/sec для модели 3.3M параметров
- Compression ratio: 474x (активации хорошо сжимаются)
- Cache benchmark: 24ms для полного прохода

### Когда использовать

✅ **Подходит для:**
- Embedded систем с ограниченной VRAM
- Виртуализированных сред (detected cache pressure)
- Обучения на CPU с OpenCL
- Прототипирования архитектур

❌ **Не подходит для:**
- Продакшена с NVIDIA GPU (лучше CUDA)
- Очень больших моделей (>100M параметров)
- Требований максимальной производительности

### Лицензия

MIT License - см. файл LICENSE
