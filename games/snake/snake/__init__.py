 # 📦 snake/__init__.py
# ================================================================
# 뱀게임 AI 패키지 초기화 파일
# ================================================================

"""
🐍 Snake Game AI Package

이 패키지는 강화학습 기반 뱀게임 AI를 구현합니다.

주요 구성요소:
- SnakeGameAI: 게임 환경 클래스
- QLearningAgent: Q-Learning 알고리즘 에이전트
- TrainingVisualizer: 학습 과정 시각화 도구
- GameColors: 다크모드 최적화 색상
- Direction: 게임 방향 열거형
"""

# 패키지 메타데이터
__version__ = "1.0.0"
__author__ = "AI/빅데이터 석사과정생"
__description__ = "강화학습 기반 뱀게임 AI 프로젝트"

# 주요 클래스들을 패키지 레벨에서 import 가능하게 설정
from .game import (
    SnakeGameAI,
    GameColors, 
    Direction,
    Point
)

from .agent import (
    QLearningAgent
)

from .train import (
    SnakeAITrainer,
    main_training,
    quick_demo, 
    advanced_training
)

from .improved_train import (
    ImprovedQLearningAgent,
    run_improved_training,
    improved_reward_function
)

# 패키지에서 export할 객체들 명시
__all__ = [
    'game',
    'SnakeGameAI',
    'GameColors',
    'Direction', 
    'Point',
    
    'agent',
    'QLearningAgent',
    
    'train',
    'SnakeAITrainer',
    'main_training',
    'quick_demo',
    'advanced_training',

    'ImprovedQLearningAgent',
    'run_improved_training',
    'improved_reward_function',
]

import os
if os.getenv('DEBUG', 'False').lower() == 'true':
    print(f"🐍 Snake AI Package v{__version__} loaded successfully!")

# 패키지 레벨 설정
DEFAULT_CONFIG = {
    'game_width': 640,
    'game_height': 480, 
    'block_size': 20,
    'game_speed': 20,
    'learning_rate': 0.1,
    'discount_factor': 0.95,
    'epsilon': 1.0
}

def get_version():
    """패키지 버전 반환"""
    return __version__

def get_default_config():
    """기본 설정 반환"""
    return DEFAULT_CONFIG.copy()

# 호환성 확인
def check_dependencies():
    """필수 라이브러리 설치 여부 확인"""
    missing = []
    
    try:
        import pygame
    except ImportError:
        missing.append('pygame 또는 pygame-ce')
    
    try:
        import numpy
    except ImportError:
        missing.append('numpy')
        
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    
    if missing:
        print(f"⚠️ 누락된 라이브러리: {', '.join(missing)}")
        return False
    else:
        print("✅ 모든 필수 라이브러리가 설치되어 있습니다!")
        return True