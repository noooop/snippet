
## install 

pip install kvikio-cu12 cupy-cuda12x


## benchmarks

### 13700k+32G*4 DDR4 3600 + 4090 + pcie 4 ssd

```
kvikio: ssd to gpu 3.0453657519158623 GB/s
kvikio: gpu to ssd 5.641049281000345 GB/s

numpy: cpu to ssd 3.4756728820261786 GB/s
numpy: ssd to cpu 6.5520971491038305 GB/s
numpy: cpu to cpu 19.668772559522562 GB/s

torch: cpu to ssd 2.120535333725255 GB/s
torch: ssd to cpu 3.6678152118238767 GB/s
torch: cpu to cpu 18.31373723387411 GB/s

torch: gpu to ssd 1.3003456617848537 GB/s
torch: ssd to gpu 2.7394149293058145 GB/s
torch: gpu to gpu 13833.578593932974 GB/s
torch: gpu to cpu 19.467226540853233 GB/s
torch: cpu to gpu 19.653485611881948 GB/s
```

### 9800X3d+32G*2 DDR5 6000 + 4090 + pcie 5 ssd
```
kvikio: ssd to gpu 4.250199702008159 GB/s
kvikio: gpu to ssd 2.025967428266533 GB/s
numpy: cpu to ssd 14.480372013212344 GB/s
numpy: ssd to cpu 15.065337143136693 GB/s
numpy: cpu to cpu 21.88959786498821 GB/s
torch: cpu to ssd 3.5873735081901086 GB/s
torch: ssd to cpu 6.99473685746602 GB/s
torch: cpu to cpu 20.497798779821196 GB/s
torch: gpu to ssd 2.131629609893504 GB/s
torch: ssd to gpu 4.732825028478302 GB/s
torch: gpu to gpu 426.55965138126726 GB/s
torch: gpu to cpu 20.967243802790705 GB/s
torch: cpu to gpu 20.8630392587053 GB/s
```

### 7800+32G*2 DDR4 3600 + V100(pcie 3 * 8) + pcie 3 ssd

```
kvikio: ssd to gpu 1.556002929824495 GB/s
kvikio: gpu to ssd 1.8874204243140742 GB/s
numpy: cpu to ssd 1.6191213667488495 GB/s
numpy: ssd to cpu 8.948630542972516 GB/s
numpy: cpu to cpu 19.909793404860856 GB/s
torch: cpu to ssd 0.7670641708131629 GB/s
torch: ssd to cpu 2.247354981032621 GB/s
torch: cpu to cpu 12.275794015769513 GB/s
torch: gpu to ssd 0.5170783927240569 GB/s
torch: ssd to gpu 1.487193273080025 GB/s
torch: gpu to gpu 376.46871031228744 GB/s
torch: gpu to cpu 6.253601889033367 GB/s
torch: cpu to gpu 6.124511468197761 GB/s
```