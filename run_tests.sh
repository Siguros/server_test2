#!/bin/bash

# 스크립트 실행 중단 시 에러 표시
set -e

echo "=== 통합 테스트 시작 ==="

# Docker 컨테이너 내에서 테스트 실행
docker exec aihwkit-ml-container bash -c "cd /workspace && python test_integration.py"

echo "=== 테스트 완료 ==="
