# flash_test

## install dependencies
```
    ./scripts/install_torch2.sh rocm # rocm
    ./scripts/install_torch2.sh cuda # cuda
```

## build with cuda

```
    export CUDA_ROOT=/path/to/cuda/root
    make cuda=1
```


## build with rocm

```
    export ROCM_ROOT=/path/to/rocm/root
    make rocm=1
```


## run
```
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/rocm/lib # required or non-default ROCM install
    ./flash_test
```