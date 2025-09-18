# 🐍 뱀게임 AI - 게임 환경 구현
# ================================================================
# 작성자: AI/빅데이터 석사과정생
# 환경: VSCode/Colab (다크모드 최적화)
# 목표: 강화학습을 위한 뱀게임 환경 구축
# ================================================================

import pygame
import numpy as np
import random
import math
from enum import Enum
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import clear_output
import time

# 게임 설정 상수들
BLOCK_SIZE = 20
SPEED = 20

# 다크모드 색상 정의 (게임 UI용)
class GameColors:
    """
    다크모드에 최적화된 게임 색상 클래스
    
    모든 색상이 어두운 배경에서 잘 보이도록 설계됨
    RGB 값으로 정의되어 Pygame에서 직접 사용 가능
    """
    # 배경 및 기본 색상
    BACKGROUND = (13, 17, 23)      # GitHub Dark Background
    GRID_LINE = (33, 38, 45)       # 격자선 (어두운 회색)
    
    # 뱀 색상
    SNAKE_HEAD = (57, 211, 83)     # 뱀 머리 (밝은 녹색)
    SNAKE_BODY = (35, 134, 54)     # 뱀 몸통 (진한 녹색)
    
    # 사과 및 UI 색상  
    APPLE = (248, 81, 73)          # 사과 (빨간색)
    TEXT_COLOR = (240, 246, 252)   # 텍스트 (흰색)
    SCORE_BG = (22, 27, 34)        # 점수 배경 (어두운 회색)
    
    # 상태 표시 색상
    DANGER = (248, 81, 73)         # 위험 (빨간색)
    SAFE = (57, 211, 83)           # 안전 (녹색)
    WARNING = (255, 223, 0)        # 경고 (노란색)

class Direction(Enum):
    """
    뱀의 이동 방향을 나타내는 열거형 클래스
    
    각 방향은 (x, y) 좌표 변화량으로 표현됨
    - RIGHT: x축 양의 방향 (+1, 0)
    - LEFT: x축 음의 방향 (-1, 0)  
    - UP: y축 음의 방향 (0, -1) - 화면 좌표계
    - DOWN: y축 양의 방향 (0, +1)
    """
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# 게임 좌표를 나타내는 네임드 튜플
Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    """
    강화학습을 위한 뱀게임 환경 클래스
    
    주요 기능:
    - 게임 상태 관리 (뱀 위치, 사과 위치, 점수 등)
    - 행동 수행 및 보상 계산
    - 게임 종료 조건 판단
    - 상태 벡터 생성 (AI 에이전트용)
    - 시각화 (선택사항)
    
    Attributes:
        w (int): 게임 화면 너비
        h (int): 게임 화면 높이  
        display: Pygame 디스플레이 객체
        clock: Pygame 시계 객체
        direction: 현재 이동 방향
        head: 뱀의 머리 위치
        snake: 뱀 전체 몸통 리스트
        score: 현재 점수
        food: 사과 위치
        frame_iteration: 현재 프레임 수
    """
    
    def __init__(self, w=640, h=480, display=True):
        """
        뱀게임 환경 초기화
        
        Args:
            w (int): 게임 화면 너비 (기본값: 640)
            h (int): 게임 화면 높이 (기본값: 480) 
            display (bool): 화면 표시 여부 (기본값: True)
            
        설명:
            - Pygame 초기화
            - 게임 화면 설정
            - 뱀과 사과 초기 위치 설정
            - 폰트 및 시계 초기화
        """
        self.w = w
        self.h = h
        self.display_enabled = display
        
        # Pygame 초기화
        if self.display_enabled:
            pygame.init()
            self.font = pygame.font.Font(None, 25)
            
            # 게임 화면 생성 (다크 테마)
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('🐍 Snake AI - 강화학습 훈련중...')
            self.clock = pygame.time.Clock()
        
        # 게임 리셋
        self.reset()
    
    def reset(self):
        """
        게임 초기화 함수
        
        Returns:
            None
            
        설명:
            - 뱀을 화면 중앙에 배치
            - 사과를 랜덤 위치에 생성
            - 점수와 프레임 수 초기화
            - 초기 방향을 오른쪽으로 설정
        """
        # 초기 방향 설정
        self.direction = Direction.RIGHT
        
        # 뱀 초기 위치 (화면 중앙)
        self.head = Point(self.w//2, self.h//2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2*BLOCK_SIZE, self.head.y)
        ]
        
        # 게임 상태 초기화
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self.get_state() 
    
    def _place_food(self):
        """
        사과를 랜덤한 위치에 배치
        
        Returns:
            None
            
        설명:
            - 뱀의 몸통과 겹치지 않는 위치에 사과 생성
            - 그리드에 맞춰 BLOCK_SIZE 단위로 배치
            - 최대 100번 시도 후 실패하면 아무 곳에나 배치
        """
        attempts = 0
        while attempts < 100:  # 무한루프 방지
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            
            # 뱀과 겹치지 않는지 확인
            if self.food not in self.snake:
                break
            attempts += 1
    
    def play_step(self, action):
        """
        한 스텝 게임을 진행하는 함수 (AI 에이전트용)
        
        Args:
            action (list): 행동 벡터 [직진, 좌회전, 우회전]
                          예: [1,0,0] = 직진, [0,1,0] = 좌회전, [0,0,1] = 우회전
        
        Returns:
            tuple: (reward, game_over, score)
                - reward (float): 이번 행동에 대한 보상
                - game_over (bool): 게임 종료 여부  
                - score (int): 현재 점수
                
        설명:
            1. 프레임 수 증가
            2. 사용자 입력 처리 (종료 등)
            3. 행동에 따라 뱀 이동
            4. 충돌 검사 및 게임 종료 판단
            5. 사과 먹기 처리
            6. 보상 계산
            7. 화면 업데이트
        """
        self.frame_iteration += 1
        
        # 1. 사용자 입력 수집 (게임 종료 등)
        if self.display_enabled:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        # 2. 행동에 따라 뱀 이동
        self._move(action)
        self.snake.insert(0, self.head)
        
        # 3. 게임 종료 조건 확인
        reward = 0
        game_over = False
        
        # 충돌 또는 시간 초과 검사
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10  # 게임 종료 시 큰 마이너스 보상
            return reward, game_over, self.score
        
        # 4. 사과 먹기 처리
        if self.head == self.food:
            self.score += 1
            reward = 10  # 사과 먹기 성공 시 큰 플러스 보상
            self._place_food()
        else:
            self.snake.pop()  # 꼬리 제거 (성장하지 않음)
        
        # 5. 거리 기반 보상 추가 (사과에 가까워질수록 보상)
        reward += self._calculate_distance_reward()
        
        # 6. 생존 보상
        reward += 0.1  # 살아있기만 해도 작은 보상
        
        # 7. 화면 업데이트
        if self.display_enabled:
            self._update_ui()
            self.clock.tick(SPEED)
        
        return reward, game_over, self.score
    
    def _calculate_distance_reward(self):
        """
        뱀 머리와 사과 사이의 거리에 기반한 보상 계산
        
        Returns:
            float: 거리 기반 보상 (-1 ~ +1)
            
        설명:
            - 사과에 가까워질수록 + 보상
            - 사과에서 멀어질수록 - 보상
            - 맨하탄 거리(Manhattan Distance) 사용
        """
        # 이전 거리 계산 (없으면 현재 거리로 초기화)
        if not hasattr(self, '_prev_distance'):
            self._prev_distance = self._get_distance_to_food()
        
        current_distance = self._get_distance_to_food()
        
        # 거리 변화에 따른 보상
        if current_distance < self._prev_distance:
            reward = 1.0  # 가까워짐
        elif current_distance > self._prev_distance:
            reward = -1.0  # 멀어짐
        else:
            reward = 0.0  # 변화 없음
        
        self._prev_distance = current_distance
        return reward
    
    def _get_distance_to_food(self):
        """
        뱀 머리와 사과 사이의 맨하탄 거리 계산
        
        Returns:
            float: 맨하탄 거리
        """
        return abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
    
    def is_collision(self, pt=None):
        """
        충돌 여부 확인
        
        Args:
            pt (Point, optional): 확인할 좌표 (기본값: 뱀의 머리)
            
        Returns:
            bool: 충돌 여부
            
        설명:
            - 벽과의 충돌 검사
            - 자기 몸통과의 충돌 검사
        """
        if pt is None:
            pt = self.head
        
        # 벽과 충돌 검사
        if (pt.x > self.w - BLOCK_SIZE or pt.x < 0 or 
            pt.y > self.h - BLOCK_SIZE or pt.y < 0):
            return True
        
        # 자기 몸통과 충돌 검사
        if pt in self.snake[1:]:
            return True
        
        return False
    
    def _move(self, action):
        """
        행동에 따라 뱀의 방향을 결정하고 이동하는 내부 함수
        
        Args:
            action (list): 행동 벡터 [직진, 좌회전, 우회전]
            
        Returns:
            None
            
        설명:
            - action[0]=1: 직진
            - action[1]=1: 현재 방향 기준 좌회전  
            - action[2]=1: 현재 방향 기준 우회전
            - 새로운 방향으로 한 칸 이동
        """
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):      # 직진
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):    # 우회전
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1] 좌회전
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        # 새로운 방향으로 이동
        x = self.head.x
        y = self.head.y
        
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        
        self.head = Point(x, y)
    
    def _update_ui(self):
        """
        게임 화면 업데이트하는
        
        Returns:
            None
            
        설명:
            - 배경 그리기
            - 뱀 그리기 (머리와 몸통 구분)
            - 사과 그리기
            - 점수 표시
            - 격자 그리기 (선택사항)
        """
        # 배경 채우기
        self.display.fill(GameColors.BACKGROUND)
        
        # 격자 그리기 (선택사항)
        self._draw_grid()
        
        # 뱀 그리기
        for i, pt in enumerate(self.snake):
            if i == 0:  # 머리
                pygame.draw.rect(self.display, GameColors.SNAKE_HEAD,
                               pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                # 머리에 눈 그리기
                eye_size = 4
                pygame.draw.circle(self.display, GameColors.BACKGROUND,
                                 (pt.x + 6, pt.y + 6), eye_size)
                pygame.draw.circle(self.display, GameColors.BACKGROUND,
                                 (pt.x + 14, pt.y + 6), eye_size)
            else:  # 몸통
                pygame.draw.rect(self.display, GameColors.SNAKE_BODY,
                               pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 사과 그리기
        pygame.draw.rect(self.display, GameColors.APPLE,
                        pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        # 점수 표시
        self._draw_score()
        
        pygame.display.flip()
    
    def _draw_grid(self):
        """
        게임 화면에 격자를 그리는 내부 함수
        
        Returns:
            None
            
        설명:
            - BLOCK_SIZE 간격으로 격자선 그리기
            - 어두운 색상으로 은은하게 표시
        """
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GameColors.GRID_LINE, 
                           (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GameColors.GRID_LINE, 
                           (0, y), (self.w, y))
    
    def _draw_score(self):
        """
        화면 상단에 점수 표시
        
        Returns:
            None
            
        설명:
            - 현재 점수와 프레임 수 표시
            - 뱀의 길이도 함께 표시
            - 다크 테마에 맞는 색상 사용
        """
        # 점수 텍스트 생성
        score_text = f"점수: {self.score}  길이: {len(self.snake)}  프레임: {self.frame_iteration}"
        text = self.font.render(score_text, True, GameColors.TEXT_COLOR)
        
        # 배경 박스 그리기
        text_rect = text.get_rect()
        bg_rect = pygame.Rect(10, 10, text_rect.width + 20, text_rect.height + 10)
        pygame.draw.rect(self.display, GameColors.SCORE_BG, bg_rect)
        pygame.draw.rect(self.display, GameColors.GRID_LINE, bg_rect, 2)
        
        # 텍스트 그리기
        self.display.blit(text, (20, 15))
    
    def get_state(self):
        """
        현재 게임 상태를 AI 에이전트용 벡터 변환
        
        Returns:
            np.array: 크기 11의 상태 벡터
            
        상태 벡터 구성:
            [0-2]: 위험 감지 (직진, 우회전, 좌회전 방향)
            [3-6]: 뱀의 이동 방향 (상/하/좌/우)  
            [7-10]: 사과의 상대적 위치 (좌/우/위/아래)
            
        설명:
            - 각 요소는 0 또는 1의 이진값
            - AI가 의사결정에 필요한 핵심 정보만 추출
            - 상태 공간을 단순화하여 학습 효율성 향상
        """
        head = self.snake[0]
        
        # 현재 방향 기준으로 좌우 방향 계산
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # 위험 감지 (직진, 우회전, 좌회전)
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),
            
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),
            
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),
            
            # 이동 방향
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 사과 위치
            self.food.x < self.head.x,  # 사과가 왼쪽에
            self.food.x > self.head.x,  # 사과가 오른쪽에
            self.food.y < self.head.y,  # 사과가 위에
            self.food.y > self.head.y   # 사과가 아래에
        ]
        
        return np.array(state, dtype=int)
    
    def close(self):
        """
        게임 종료 및 리소스 정리
        
        Returns:
            None
        """
        if self.display_enabled:
            pygame.quit()

# ================================================================
# 테스트 및 시연용 함수들
# ================================================================

def demo_game():
    """
    게임 환경 테스트 - 데모 함수
    
    Returns:
        None
        
    설명:
        - 수동으로 게임을 플레이하여 환경 테스트
        - 화살표 키로 조작 가능
        - AI 학습 전에 게임이 정상 작동하는지 확인
    """
    print("🎮 뱀게임 데모 시작!")
    print("화살표 키로 조작하세요. ESC로 종료.")
    
    game = SnakeGameAI()
    
    while True:
        # 랜덤 행동으로 테스트
        action = [0, 0, 0]
        action[random.randint(0, 2)] = 1
        
        reward, game_over, score = game.play_step(action)
        
        if game_over:
            print(f"게임 종료! 최종 점수: {score}")
            break
    
    game.close()

def test_state_representation():
    """
    상태 표현이 올바른지 테스트
    
    Returns:
        None
        
    설명:
        - 다양한 게임 상황에서 상태 벡터 출력
        - 상태 벡터의 각 요소가 올바른 정보를 나타내는지 확인
    """
    print("🧪 상태 표현 테스트")
    print("="*50)
    
    game = SnakeGameAI(display=False)  # 화면 없이 테스트
    
    for i in range(5):
        state = game.get_state()
        print(f"스텝 {i+1}: {state}")
        print(f"뱀 위치: {game.head}, 사과 위치: {game.food}")
        print(f"점수: {game.score}, 뱀 길이: {len(game.snake)}")
        print("-" * 30)
        
        # 랜덤 행동
        action = [0, 0, 0]
        action[random.randint(0, 2)] = 1
        
        reward, game_over, score = game.play_step(action)
        if game_over:
            print("게임 종료!")
            break
    
    game.close()

if __name__ == "__main__":
    print("🐍 뱀게임 AI 환경이 준비되었습니다!")
    print("\n선택하세요:")
    print("1. demo_game() - 게임 데모 실행")
    print("2. test_state_representation() - 상태 표현 테스트")
    
    # 데모 실행 (주석 해제하여 사용)
    demo_game()
    test_state_representation()