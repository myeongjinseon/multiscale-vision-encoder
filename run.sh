#!/bin/bash
# ================================================================
#  Multi-Scale Vision Encoder — 실행 스크립트
#  Usage: bash run.sh [command] [options]
#
#  Commands:
#    setup       환경 세팅 (패키지 설치)
#    test        빠른 파이프라인 테스트 (mock 데이터, 2 epoch)
#    train       전체 학습 (captioning + grounding, multiscale + baseline)
#    ablation    Ablation study 실행
#    eval        학습된 모델 평가
#    viz         시각화 생성
#    all         setup → test → train → ablation → eval → viz 순차 실행
#
#  Options:
#    --gpu ID    사용할 GPU 번호 지정 (기본: 자동 감지)
#                단일 GPU:  --gpu 0
#                특정 GPU:  --gpu 2
#                멀티 GPU:  --gpu 0,1,2,3
#                CPU 강제:  --gpu cpu
#
#  Examples:
#    bash run.sh setup                # 처음 한 번만
#    bash run.sh test                 # 파이프라인 동작 확인
#    bash run.sh train --gpu 0        # GPU 0번으로 학습
#    bash run.sh train --gpu 2,3      # GPU 2,3번 사용
#    bash run.sh ablation --gpu 1     # GPU 1번으로 ablation
#    bash run.sh all --gpu 0          # 전부 GPU 0번으로
# ================================================================

set -e  # 에러 발생 시 중단

# ── 옵션 파싱 ──
GPU_ARG=""
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_ARG="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

# ── 경로 설정 ──
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/default.yaml"
EXP_DIR="experiments"
DEVICE="cpu"

# GPU 설정
if [ "$GPU_ARG" = "cpu" ]; then
    # CPU 강제 사용
    DEVICE="cpu"
    export CUDA_VISIBLE_DEVICES=""
elif [ -n "$GPU_ARG" ]; then
    # 사용자가 GPU 번호 지정
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
    DEVICE="cuda"
else
    # 자동 감지
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
    fi
fi

# ── 색상 출력 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'  # No Color

banner() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  $1${NC}"
    if [ "$DEVICE" = "cuda" ]; then
        echo -e "${CYAN}  GPU: ${CUDA_VISIBLE_DEVICES:-auto}${NC}"
    else
        echo -e "${CYAN}  Device: CPU${NC}"
    fi
    echo -e "${CYAN}════════════════════════════════════════════════════${NC}"
    echo ""
}

success() { echo -e "${GREEN}✅ $1${NC}"; }
info()    { echo -e "${BLUE}ℹ️  $1${NC}"; }
warn()    { echo -e "${YELLOW}⚠️  $1${NC}"; }
fail()    { echo -e "${RED}❌ $1${NC}"; exit 1; }

# ================================================================
#  1. SETUP — 환경 세팅
# ================================================================
cmd_setup() {
    banner "환경 세팅"

    info "Python 버전 확인..."
    python --version || fail "Python이 설치되어 있지 않습니다"

    info "필수 패키지 설치..."
    pip install torch torchvision pyyaml matplotlib pillow --quiet

    info "(선택) Pretrained backbone 패키지 설치..."
    pip install open_clip_torch timm --quiet 2>/dev/null || \
        warn "open_clip/timm 설치 실패 — mock backbone으로 대체됩니다"

    info "프로젝트 구조 확인..."
    for dir in models data utils scripts configs visualization; do
        if [ ! -d "$dir" ]; then
            fail "$dir/ 디렉토리가 없습니다"
        fi
    done

    mkdir -p "$EXP_DIR"
    mkdir -p data/coco data/refcoco

    info "Device: $DEVICE"
    if [ "$DEVICE" = "cuda" ]; then
        info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
        python -c "
import torch
n = torch.cuda.device_count()
print(f'  사용 가능 GPU: {n}개')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'    GPU {i}: {name} ({mem:.1f} GB)')
" 2>/dev/null || true
    fi
    success "환경 세팅 완료!"
}

# ================================================================
#  2. TEST — 빠른 파이프라인 테스트
# ================================================================
cmd_test() {
    banner "파이프라인 테스트 (Mock 데이터, 2 Epochs)"

    info "모듈 임포트 테스트..."
    python -c "
import sys; sys.path.insert(0, '.')
from models import build_encoder, CaptioningHead, GroundingHead, load_config
from data import build_coco_dataloader, build_refcoco_dataloader
from utils import CaptioningEvaluator, GroundingEvaluator
print('모든 모듈 임포트 성공')
" || fail "모듈 임포트 실패"

    info "Forward pass 테스트..."
    python -c "
import torch, sys; sys.path.insert(0, '.')
from models import build_encoder, CaptioningHead, GroundingHead, load_config

config = load_config('configs/default.yaml')
encoder = build_encoder(config)
dummy = torch.randn(2, 3, 224, 224)
out = encoder(dummy, return_intermediate=True)

# Captioning
cap = CaptioningHead(visual_dim=encoder.output_dim)
cap_out = cap(out['features'], torch.randint(1,1000,(2,20)), torch.ones(2,20,dtype=torch.long))

# Grounding
gnd = GroundingHead(visual_dim=encoder.output_dim)
gnd_out = gnd(out['patch_features'], torch.randint(1,1000,(2,15)), torch.ones(2,15,dtype=torch.long), torch.rand(2,4))

print(f'Encoder:    {out[\"features\"].shape}')
print(f'Captioning: loss={cap_out[\"loss\"].item():.4f}')
print(f'Grounding:  loss={gnd_out[\"loss\"].item():.4f}')
print('Forward pass 성공!')
" || fail "Forward pass 실패"

    info "Mini training 테스트 (Captioning, 2 epochs)..."
    python scripts/train_captioning.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --exp-name "test_captioning" \
        2>&1 | python -c "
import sys
# default.yaml은 30 epoch이므로 테스트용으로 빨리 끊기
for i, line in enumerate(sys.stdin):
    print(line, end='')
    if i > 100:
        break
" &
    TEST_PID=$!
    sleep 10 && kill $TEST_PID 2>/dev/null
    wait $TEST_PID 2>/dev/null || true

    success "파이프라인 테스트 통과!"
    info "본격 학습은 'bash run.sh train' 으로 실행하세요"
}

# ================================================================
#  3. TRAIN — 전체 학습
# ================================================================
cmd_train() {
    banner "전체 학습 시작"
    info "Device: $DEVICE"

    # --- Captioning: Multi-scale ---
    echo ""
    info "[1/4] Captioning — Multi-Scale Encoder"
    python scripts/train_captioning.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --exp-name "captioning_multiscale"
    success "Captioning (multi-scale) 완료"

    # --- Captioning: Baseline ---
    echo ""
    info "[2/4] Captioning — Baseline (Single-Scale)"
    python scripts/train_captioning.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --exp-name "captioning_baseline" \
        --baseline
    success "Captioning (baseline) 완료"

    # --- Grounding: Multi-scale ---
    echo ""
    info "[3/4] Grounding — Multi-Scale Encoder"
    python scripts/train_grounding.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --exp-name "grounding_multiscale"
    success "Grounding (multi-scale) 완료"

    # --- Grounding: Baseline ---
    echo ""
    info "[4/4] Grounding — Baseline (Single-Scale)"
    python scripts/train_grounding.py \
        --config "$CONFIG" \
        --device "$DEVICE" \
        --exp-name "grounding_baseline" \
        --baseline
    success "Grounding (baseline) 완료"

    banner "전체 학습 완료!"
    info "결과: $EXP_DIR/"
    ls -d "$EXP_DIR"/*/ 2>/dev/null
}

# ================================================================
#  4. ABLATION — Ablation Study
# ================================================================
cmd_ablation() {
    banner "Ablation Study"

    ABLATION_TASK="${1:-grounding}"
    info "Task: $ABLATION_TASK | Device: $DEVICE"

    # A1: 스케일 수
    echo ""
    info "[A1] 스케일 수 (1 → 4)"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter A1 \
        --quick
    success "A1 완료"

    # A2: Fusion 방법
    echo ""
    info "[A2] Fusion 방법 비교"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter A2 \
        --quick
    success "A2 완료"

    # A3: Residual Refinement
    echo ""
    info "[A3] Residual Refinement 유무"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter A3 \
        --quick
    success "A3 완료"

    # A4: Backbone
    echo ""
    info "[A4] Backbone 비교"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter A4 \
        --quick
    success "A4 완료"

    # A5: Layer 선택
    echo ""
    info "[A5] Layer 선택 전략"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter A5 \
        --quick
    success "A5 완료"

    # Baseline
    echo ""
    info "[Baseline] Single-Scale"
    python scripts/run_ablations.py \
        --config "$CONFIG" \
        --task "$ABLATION_TASK" \
        --device "$DEVICE" \
        --filter baseline \
        --quick
    success "Baseline 완료"

    banner "Ablation Study 전체 완료!"
    info "결과 JSON/이미지: $EXP_DIR/ablation_*/"
}

# ================================================================
#  5. EVAL — 평가
# ================================================================
cmd_eval() {
    banner "모델 평가"

    # Captioning 평가
    CKPT_CAP="$EXP_DIR/captioning_multiscale/checkpoints/best_model.pt"
    if [ -f "$CKPT_CAP" ]; then
        info "Captioning 평가 중..."
        python scripts/evaluate.py \
            --checkpoint "$CKPT_CAP" \
            --task captioning \
            --device "$DEVICE"
        success "Captioning 평가 완료"
    else
        warn "Captioning 체크포인트 없음: $CKPT_CAP"
        warn "'bash run.sh train' 을 먼저 실행하세요"
    fi

    # Grounding 평가
    CKPT_GND="$EXP_DIR/grounding_multiscale/checkpoints/best_model.pt"
    if [ -f "$CKPT_GND" ]; then
        info "Grounding 평가 중..."
        python scripts/evaluate.py \
            --checkpoint "$CKPT_GND" \
            --task grounding \
            --device "$DEVICE"
        success "Grounding 평가 완료"
    else
        warn "Grounding 체크포인트 없음: $CKPT_GND"
    fi
}

# ================================================================
#  6. VIZ — 시각화
# ================================================================
cmd_viz() {
    banner "시각화 생성"

    CKPT_GND="$EXP_DIR/grounding_multiscale/checkpoints/best_model.pt"
    if [ -f "$CKPT_GND" ]; then
        info "Attention 시각화 생성 중..."
        python visualization/visualize_attention.py \
            --checkpoint "$CKPT_GND" \
            --task grounding \
            --num-samples 20 \
            --output "$EXP_DIR/visualizations" \
            --compare-baseline \
            --device "$DEVICE"
        success "시각화 완료: $EXP_DIR/visualizations/"
    else
        warn "체크포인트 없음: $CKPT_GND"
        warn "'bash run.sh train' 을 먼저 실행하세요"
    fi
}

# ================================================================
#  7. ALL — 전체 파이프라인
# ================================================================
cmd_all() {
    banner "전체 파이프라인 실행"
    START_TIME=$(date +%s)

    cmd_setup
    cmd_test
    cmd_train
    cmd_ablation
    cmd_eval
    cmd_viz

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    HOURS=$(( ELAPSED / 3600 ))
    MINS=$(( (ELAPSED % 3600) / 60 ))

    banner "전체 파이프라인 완료! (${HOURS}h ${MINS}m)"
    info "결과 디렉토리:"
    echo ""
    find "$EXP_DIR" -maxdepth 2 -type d | head -20
}

# ================================================================
#  HELP
# ================================================================
cmd_help() {
    echo ""
    echo "Usage: bash run.sh [command] [--gpu ID]"
    echo ""
    echo "Commands:"
    echo "  setup      환경 세팅 (패키지 설치)"
    echo "  test       빠른 파이프라인 테스트 (mock 데이터)"
    echo "  train      전체 학습 (captioning + grounding)"
    echo "  ablation   Ablation study (A1~A5 + baseline)"
    echo "  eval       학습된 모델 평가"
    echo "  viz        시각화 생성"
    echo "  all        전부 순차 실행"
    echo ""
    echo "GPU Options:"
    echo "  --gpu 0        GPU 0번만 사용"
    echo "  --gpu 2        GPU 2번만 사용"
    echo "  --gpu 0,1,2,3  GPU 0~3번 사용 (멀티 GPU)"
    echo "  --gpu cpu      CPU 강제 사용"
    echo "  (생략)          자동 감지"
    echo ""
    echo "Examples:"
    echo "  bash run.sh setup              # 처음 한 번"
    echo "  bash run.sh test --gpu 0       # GPU 0번으로 테스트"
    echo "  bash run.sh train --gpu 0      # GPU 0번으로 학습"
    echo "  bash run.sh train --gpu 2,3    # GPU 2,3번 사용"
    echo "  bash run.sh ablation --gpu 1   # GPU 1번으로 ablation"
    echo "  bash run.sh all --gpu 0        # 전부 GPU 0번"
    echo ""
}

# ================================================================
#  MAIN
# ================================================================
COMMAND="${1:-help}"

case "$COMMAND" in
    setup)    cmd_setup ;;
    test)     cmd_test ;;
    train)    cmd_train ;;
    ablation) cmd_ablation "${2}" ;;
    eval)     cmd_eval ;;
    viz)      cmd_viz ;;
    all)      cmd_all ;;
    help|*)   cmd_help ;;
esac
