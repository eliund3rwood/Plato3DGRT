#!/bin/bash
set -e

WITH_GCC11=false
if [ $# -ge 1 ]; then
    if [ "$1" = "WITH_GCC11" ]; then
        WITH_GCC11=true
    fi
fi

CUDA_VERSION=${CUDA_VERSION:-"12.8.1"}

echo "Arguments:"
echo "  WITH_GCC11: $WITH_GCC11"
echo "  CUDA_VERSION: $CUDA_VERSION"
echo ""

if [ "$CUDA_VERSION" != "12.8.1" ]; then
    echo "Unsupported CUDA version: $CUDA_VERSION (only 12.8.1 supported in venv mode)"
    exit 1
fi

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0;10.0;12.0+PTX"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if ! command -v python >/dev/null; then
    echo "Python not found"
    exit 1
fi

if ! command -v pip >/dev/null; then
    echo "pip not found"
    exit 1
fi

if [ "$WITH_GCC11" = true ]; then
    if ! command -v gcc-11 >/dev/null; then
        echo "gcc-11 not found"
        exit 1
    fi
    if ! command -v g++-11 >/dev/null; then
        echo "g++-11 not found"
        exit 1
    fi
    export CC=$(which gcc-11)
    export CXX=$(which g++-11)
fi

gcc_version=$(gcc -dumpversion | cut -d '.' -f 1)
if [ "$gcc_version" -gt 11 ]; then
    echo "gcc > 11 detected. CUDA extension builds may fail."
fi

pip install --upgrade pip wheel
pip install "setuptools<70"

# install PyTorch matching system CUDA 12.8
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

pip install --force-reinstall "numpy<2"

# build kaolin from source (required for CUDA 12.x)
rm -rf thirdparty/kaolin
git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git thirdparty/kaolin

pushd thirdparty/kaolin
git checkout c2da967b9e0d8e3ebdbd65d3e8464d7e39005203

sed -i 's!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type()!AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.scalar_type()!g' \
    kaolin/csrc/render/spc/raytrace_cuda.cu

pip install ninja imageio imageio-ffmpeg

pip install \
    -r tools/viz_requirements.txt \
    -r tools/requirements.txt \
    -r tools/build_requirements.txt

IGNORE_TORCH_VER=1 python setup.py install
popd

rm -rf thirdparty/kaolin

git submodule update --init --recursive

pip install --no-build-isolation -r requirements.txt
pip install --no-build-isolation -e .

echo "Setup completed successfully (venv mode, CUDA 12.8)."