#!/bin/bash

# 스크립트 실행 중단 시 에러 표시
set -e

echo "=== aihwkit-ml 도커 이미지 빌드 시작 ==="

# NVIDIA Container Toolkit 확인
if ! command -v nvidia-smi &> /dev/null; then
    echo "경고: NVIDIA Container Toolkit이 설치되어 있지 않습니다."
    echo "CUDA 지원이 필요한 경우 다음 링크를 참조하여 설치하세요:"
    echo "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Docker 이미지 빌드
echo "Docker 이미지 빌드 중..."
docker build -t aihwkit-ml \
    --build-arg CUDA_VERSION=11.8.0 \
    --build-arg USERNAME=$(whoami) \
    --build-arg USERID=$(id -u) \
    --build-arg GROUPID=$(id -g) \
    --build-arg CUDA_ARCH="70;75;80;86" \
    .

echo "=== 빌드 완료 ==="

# 컨테이너 실행
echo "=== 컨테이너 실행 ==="
docker run --gpus all -it \
    --name aihwkit-ml-container \
    -v $(pwd)/ml:/workspace/ml \
    -v $(pwd)/aihwkit:/workspace/aihwkit \
    -w /workspace \
    aihwkit-ml

# 테스트 스크립트 실행
echo "=== 환경 테스트 실행 ==="
docker exec aihwkit-ml-container bash -c "
echo '테스트 1: CUDA 가용성 확인'
python3 -c 'import torch; print(\"CUDA available:\", torch.cuda.is_available())'

echo '테스트 2: aihwkit CUDA 지원 확인'
python3 -c 'from aihwkit.simulator.rpu_base import cuda; print(\"aihwkit CUDA compiled:\", cuda.is_compiled())'

echo '테스트 3: PyTorch Lightning 설치 확인'
python3 -c 'import pytorch_lightning as pl; print(\"PyTorch Lightning version:\", pl.__version__)'

echo '테스트 4: ML 저장소 import 테스트'
python3 -c 'import sys; print(\"Python path:\", sys.path)'
"

echo "=== 설정 완료 ==="
echo "컨테이너에 접속하려면 다음 명령어를 실행하세요:"
echo "docker exec -it aihwkit-ml-container bash"
