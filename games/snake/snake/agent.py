# 🧠 뱀게임 AI - Q-Learning 에이전트 구현

import numpy as np
import random
import pickle
import os
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# 다크모드 시각화 설정
plt.style.use('dark_background')

class QLearningAgent:
    """
    Q-Learning 알고리즘을 사용하는 강화학습 에이전트 클래스
    
    Q-Learning 핵심 원리:
    - Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
    - s: 현재 상태, a: 현재 행동, r: 보상
    - s': 다음 상태, α: 학습률, γ: 할인인수
    
    Attributes:
        lr (float): 학습률 (Learning Rate)
        gamma (float): 할인인수 (Discount Factor)  
        epsilon (float): 탐험 확률 (Exploration Rate)
        epsilon_min (float): 최소 탐험 확률
        epsilon_decay (float): 탐험 확률 감소율
        q_table (defaultdict): Q-테이블 (상태-행동 쌍의 Q값 저장)
        memory (deque): 최근 경험 저장소
        training_history (dict): 학습 진행 기록
    """
    
    def __init__(self, lr=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_min=0.01, epsilon_decay=0.995):
        """
        Q-Learning 에이전트 초기화
        
        Args:
            lr (float): 학습률 - Q값 업데이트 속도 조절 (기본값: 0.1)
            gamma (float): 할인인수 - 미래 보상 중요도 (기본값: 0.95)
            epsilon (float): 초기 탐험 확률 (기본값: 1.0 = 100% 탐험)
            epsilon_min (float): 최소 탐험 확률 (기본값: 0.01 = 1% 탐험)
            epsilon_decay (float): 탐험 확률 감소율 (기본값: 0.995)
            
        설명:
            - 높은 학습률: 빠른 학습, 하지만 불안정할 수 있음
            - 높은 할인인수: 장기적 보상 중시
            - 높은 탐험 확률: 초기엔 많이 탐험, 점진적으로 감소
        """
        # 하이퍼파라미터 설정
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-테이블 초기화 (defaultdict로 자동 0 초기화)
        self.q_table = {}
        
        # 메모리 및 기록 초기화
        self.memory = deque(maxlen=10000)  # 최근 10000개 경험만 저장
        self.training_history = {
            'scores': [],
            'rewards': [],
            'epsilons': [],
            'q_table_size': [],
            'avg_q_values': []
        }
        
        print("🧠 Q-Learning 에이전트가 초기화되었습니다!")
        print(f"📊 하이퍼파라미터: lr={lr}, gamma={gamma}, epsilon={epsilon}")
    
    def get_state_key(self, state):
        """
        상태 배열을 Q-테이블의 키로 변환하는 함수
        
        Args:
            state (np.array): 게임 상태 벡터 (크기 11)
            
        Returns:
            str: Q-테이블에서 사용할 문자열 키
            
        설명:
            - NumPy 배열은 딕셔너리 키로 사용할 수 없음
            - 배열을 문자열로 변환하여 키로 사용
            - 예: [1,0,0,1,0,0,0,1,0,0,0] -> "1,0,0,1,0,0,0,1,0,0,0"
        """
        return ','.join(map(str, state.astype(int)))
    
    def get_action(self, state):
        """
        현재 상태에서 행동을 선택하는 함수 (Epsilon-Greedy 정책)
        
        Args:
            state (np.array): 현재 게임 상태
            
        Returns:
            np.array: 선택된 행동 벡터 [직진, 우회전, 좌회전]
            
        설명:
            Epsilon-Greedy 정책:
            - epsilon 확률로 랜덤 행동 (탐험, Exploration)
            - (1-epsilon) 확률로 최적 행동 (활용, Exploitation)
            - 학습 초기: 높은 탐험, 학습 후기: 높은 활용
        """
        state_key = self.get_state_key(state)
        
        # Epsilon-Greedy 정책으로 행동 선택
        if random.random() < self.epsilon:
            # 탐험: 랜덤 행동 선택
            action_idx = random.randint(0, 2)
        else:
            # 활용: Q값이 가장 높은 행동 선택
            q_values = self.q_table[state_key]
            action_idx = np.argmax(q_values)
        
        # 행동을 원-핫 벡터로 변환
        action = np.zeros(3)
        action[action_idx] = 1
        
        return action
    
    def remember(self, state, action, reward, next_state, done):
        """
        경험을 메모리에 저장하는 함수
        
        Args:
            state (np.array): 현재 상태
            action (np.array): 수행한 행동
            reward (float): 받은 보상
            next_state (np.array): 다음 상태  
            done (bool): 에피소드 종료 여부
            
        Returns:
            None
            
        설명:
            - 강화학습의 핵심인 (s, a, r, s', done) 튜플 저장
            - 나중에 Q값 업데이트에 사용
            - deque 자료구조로 최신 경험만 유지
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning 알고리즘으로 Q값을 업데이트하는 함수
        
        Args:
            state (np.array): 현재 상태
            action (np.array): 수행한 행동
            reward (float): 받은 보상
            next_state (np.array): 다음 상태
            done (bool): 에피소드 종료 여부
            
        Returns:
            None
            
        설명:
            Q-Learning 업데이트 공식:
            Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
            
            - 현재 Q값과 목표값의 차이만큼 업데이트
            - 목표값 = 즉시보상 + 할인된 미래 최대 Q값
            - 에피소드 종료 시 미래 보상은 0
        """
        # 상태와 행동을 키로 변환
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_idx = np.argmax(action)
        
        # 현재 Q값
        current_q = self.q_table[state_key][action_idx]
        
        # 목표 Q값 계산
        if done:
            # 게임 종료 시 미래 보상 없음
            target_q = reward
        else:
            # 다음 상태에서 최대 Q값
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * max_next_q
        
        # Q값 업데이트 (Q-Learning 공식)
        self.q_table[state_key][action_idx] += self.lr * (target_q - current_q)
        
        # 탐험 확률 감소 (점진적으로 활용 위주로 변경)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def batch_learn(self, batch_size=32):
        """
        메모리에서 배치를 샘플링하여 한 번에 학습하는 함수
        
        Args:
            batch_size (int): 배치 크기 (기본값: 32)
            
        Returns:
            None
            
        설명:
            - 메모리에 충분한 경험이 쌓이면 배치 학습 수행
            - 랜덤 샘플링으로 데이터 상관관계 제거
            - 학습 안정성 향상
        """
        if len(self.memory) < batch_size:
            return
        
        # 랜덤하게 배치 샘플링
        batch = random.sample(self.memory, batch_size)
        
        # 배치의 각 경험에 대해 학습
        for state, action, reward, next_state, done in batch:
            self.learn(state, action, reward, next_state, done)
    
    def get_q_table_stats(self):
        """
        Q-테이블의 통계 정보를 반환하는 함수
        
        Returns:
            dict: Q-테이블 통계 정보
            
        반환 정보:
            - size: Q-테이블 크기 (상태 수)
            - avg_q_value: 평균 Q값
            - max_q_value: 최대 Q값
            - min_q_value: 최소 Q값
            - explored_states: 탐험한 상태 수
        """
        if not self.q_table:
            return {
                'size': 0,
                'avg_q_value': 0,
                'max_q_value': 0,
                'min_q_value': 0,
                'explored_states': 0
            }
        
        all_q_values = []
        for q_values in self.q_table.values():
            all_q_values.extend(q_values)
        
        all_q_values = np.array(all_q_values)
        
        return {
            'size': len(self.q_table),
            'avg_q_value': np.mean(all_q_values),
            'max_q_value': np.max(all_q_values),
            'min_q_value': np.min(all_q_values),
            'explored_states': len(self.q_table)
        }
    
    def save_model(self, filepath):
        """
        학습된 Q-테이블과 에이전트 상태를 저장하는 함수
        
        Args:
            filepath (str): 저장할 파일 경로
            
        Returns:
            None
            
        설명:
            - Q-테이블, 하이퍼파라미터, 학습 기록을 모두 저장
            - pickle 형식으로 저장하여 나중에 로드 가능
            - 모델 버전과 저장 시간도 함께 기록
        """
        model_data = {
            'q_table': dict(self.q_table),  # defaultdict을 일반 dict로 변환
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'training_history': self.training_history,
            'save_time': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 모델 저장
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 모델이 저장되었습니다: {filepath}")
        print(f"📊 Q-테이블 크기: {len(self.q_table)}개 상태")
    
    def load_model(self, filepath):
        """
        저장된 Q-테이블과 에이전트 상태를 로드하는 함수
        
        Args:
            filepath (str): 로드할 파일 경로
            
        Returns:
            bool: 로드 성공 여부
            
        설명:
            - 이전에 저장된 모델을 불러와서 학습 재개 가능
            - 모든 하이퍼파라미터와 학습 기록도 복원
        """
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Q-테이블 복원
            self.q_table = defaultdict(lambda: np.zeros(3))
            self.q_table.update(model_data['q_table'])
            
            # 하이퍼파라미터 복원
            self.lr = model_data['lr']
            self.gamma = model_data['gamma']
            self.epsilon = model_data['epsilon']
            self.epsilon_min = model_data['epsilon_min']
            self.epsilon_decay = model_data['epsilon_decay']
            
            # 학습 기록 복원
            self.training_history = model_data['training_history']
            
            print(f"📂 모델이 로드되었습니다: {filepath}")
            print(f"📊 Q-테이블 크기: {len(self.q_table)}개 상태")
            print(f"🎯 현재 Epsilon: {self.epsilon:.4f}")
            
            return True
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {filepath}")
            return False
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def update_training_history(self, score, total_reward, episode):
        """
        학습 진행 상황을 기록하는 함수
        
        Args:
            score (int): 에피소드 점수
            total_reward (float): 에피소드 총 보상
            episode (int): 에피소드 번호
            
        Returns:
            None
            
        설명:
            - 에피소드별 성과를 기록하여 학습 분석에 활용
            - 나중에 그래프로 시각화할 데이터 축적
        """
        self.training_history['scores'].append(score)
        self.training_history['rewards'].append(total_reward)
        self.training_history['epsilons'].append(self.epsilon)
        
        # Q-테이블 통계 기록
        stats = self.get_q_table_stats()
        self.training_history['q_table_size'].append(stats['size'])
        self.training_history['avg_q_values'].append(stats['avg_q_value'])
    
    def get_best_action_for_state(self, state):
        """
        주어진 상태에서 최적 행동을 반환하는 함수 (탐험 없이)
        
        Args:
            state (np.array): 게임 상태
            
        Returns:
            tuple: (최적 행동 벡터, Q값 배열)
            
        설명:
            - 테스트나 실제 플레이 시 사용
            - 탐험 없이 순수하게 학습된 정책만 사용
        """
        state_key = self.get_state_key(state)
        q_values = self.q_table[state_key]
        
        # 최적 행동 선택
        action_idx = np.argmax(q_values)
        action = np.zeros(3)
        action[action_idx] = 1
        
        return action, q_values

# ================================================================
# 테스트 함수
# ================================================================

def test_agent():
    """
    Q-Learning 에이전트 기본 기능 테스트
    
    Returns:
        None
        
    설명:
        - 에이전트 초기화
        - 기본 메서드들 동작 확인
        - 간단한 학습 시뮬레이션
    """
    print("🧪 Q-Learning 에이전트 테스트 시작")
    print("="*50)
    
    # 에이전트 생성
    agent = QLearningAgent()
    
    # 가상의 상태와 행동으로 테스트
    test_state = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1])
    
    print(f"🔍 테스트 상태: {test_state}")
    
    # 행동 선택 테스트
    action = agent.get_action(test_state)
    print(f"🎯 선택된 행동: {action}")
    
    # 학습 테스트
    next_state = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    reward = 5.0
    done = False
    
    agent.learn(test_state, action, reward, next_state, done)
    print(f"📚 학습 완료 - 보상: {reward}")
    
    # Q-테이블 통계
    stats = agent.get_q_table_stats()
    print(f"📊 Q-테이블 통계: {stats}")
    
    print("✅ 테스트 완료!")

if __name__ == "__main__":
    print("🧠 Q-Learning 에이전트가 준비되었습니다!")
    print("\n테스트 실행:")
    test_agent()