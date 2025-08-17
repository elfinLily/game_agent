
__version__ = "1.0.0"
__author__ = "AI/빅데이터 석사과정"

#from . import breakout
#from . import flappy_bird
from .snake import (agent, game, train)
#from . import tetris

# 하위 게임 모듈들을 임포트 가능하도록 설정
__all__ = [
    "agent",      # 스네이크 게임
    "game",
    "train",
    # "breakout",   # 벽돌깨기 게임
    # "flappy_bird", # 플래피 버드 게임
    # "tetris"      # 테트리스 게임
]