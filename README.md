# flash_test

Accuracy test for `at::_flash_attention_forward` against a modified [reference attention function](https://github.com/Dao-AILab/flash-attention/blob/0e79d71175346c7151f49ab6287084a052bc9613/tests/test_flash_attn.py#L217).

## Install Dependencies
```
./scripts/install_torch2.sh rocm
# OR
./scripts/install_torch2.sh cuda
```

## Build with CUDA

```
export CUDA_ROOT=/path/to/cuda/root # if not using default
make cuda=1
```


## Build with ROCM

```
export ROCM_ROOT=/path/to/rocm/root # if not using default
make rocm=1
```


## Run
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/rocm/lib # if non-default ROCM install
./flash_test
```
