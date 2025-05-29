# flash_test

## install dependencies
```
./scripts/install_torch2.sh rocm
# OR
./scripts/install_torch2.sh cuda
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
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/rocm/lib # if non-default ROCM install
./flash_test
```