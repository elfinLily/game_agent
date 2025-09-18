#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
games/snake/improved_train.py - 성능 개선된 Snake AI 훈련

기존 train.py의 성능 문제를 해결한 개선된 버전
평균 점수 0.7점 → 10+점으로 성능 향상 목표

Author: 조주은 Lily
Created: 2025-08-05
"""

import os
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# 상위 디렉토리의 모듈들 임포트를 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from games.snake.game import SnakeGameAI
from games.snake.agent import QLearningAgent

# 다크모드 친화적 컬러 코드
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

class ImprovedQLearningAgent(QLearningAgent):
    """성능 개선된 Q-Learning 에이전트"""
    
    def __init__(self, 
                 learning_rate=0.7,        # 기존 0.3 → 0.7로 대폭 증가
                 discount_factor=0.99,     # 기존 0.95 → 0.99로 증가  
                 epsilon=0.9,              # 기존 0.8 → 0.9로 증가 (초기 탐험 극대화)
                 epsilon_decay=0.9995,     # 천천히 감소
                 epsilon_min=0.1):         # 최소값을 높게 유지
        
        # super().__init__() 호출 전에 속성을 먼저 설정합니다.
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        
        print(f"{Colors.GREEN}🚀 개선된 에이전트 초기화{Colors.RESET}")
        print(f"   학습률: {self.learning_rate} (기존 대비 +133%)")
        print(f"   할인인자: {self.discount_factor} (기존 대비 +4%)")
        print(f"   초기 탐험율: {self.epsilon} (기존 대비 +12.5%)")
        print(f"   최소 탐험율: {self.epsilon_min} (기존 대비 +1000%)")
        # 부모 클래스의 __init__을 호출하여 update_q_table과 같은 메서드를 상속받습니다.
        super().__init__(learning_rate, discount_factor, epsilon, epsilon_decay, epsilon_min)
    
    def get_action(self, state):
        """개선된 행동 선택 - 더 적극적인 탐험"""
        state_key = str(state)
        
        # Q-테이블 초기화 (더 넓은 범위의 초기값)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.random.uniform(-2, 2, 3)  # 기존 [-1,1] → [-2,2]
        
        # Epsilon-greedy with better exploration
        if np.random.random() < self.epsilon:
            # 완전 랜덤이 아닌 스마트한 탐험
            if np.random.random() < 0.3:  # 30%는 완전 랜덤
                return np.random.randint(0, 3)
            else:  # 70%는 Q값이 낮은 행동 우선 (underexplored actions)
                q_values = self.q_table[state_key]
                # 가장 적게 시도된 행동에 약간의 우선권
                return np.argmin(q_values) if np.random.random() < 0.5 else np.random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state_key])

def improved_reward_function(game, prev_score, curr_score, done, steps):
    """대폭 개선된 보상 함수"""
    reward = 0
    
    # 1. 점수 증가시 큰 보상
    score_diff = curr_score - prev_score
    if score_diff > 0:
        reward += score_diff * 20  # 기존 10 → 20으로 증가
        print(f"🍎 먹이 획득! 보상: +{score_diff * 20}")
    
    # 2. 먹이와의 거리 기반 보상 (핵심 개선사항)
    if hasattr(game, 'head') and hasattr(game, 'food'):
        head_x, head_y = game.head.x, game.head.y
        food_x, food_y = game.food.x, game.food.y
        
        # 맨하탄 거리 계산
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        max_distance = (game.w // 20) + (game.h // 20)  # 최대 가능 거리 (블록 단위)
        
        # 거리에 반비례하는 보상 (가까울수록 큰 보상)
        distance_reward = (max_distance - distance) / max_distance * 2
        reward += distance_reward
        
        # 매우 가까우면 추가 보상
        if distance <= 2:
            reward += 5
        elif distance <= 4:
            reward += 2
        elif distance <= 6:
            reward += 1
    
    # 3. 생존 보상 (생존 자체에 의미 부여)
    reward += 0.2  # 기존 0.1 → 0.2
    
    # 4. 충돌시 큰 페널티
    if done:
        reward = -50  # 기존 -10 → -50으로 증가
        print(f"💀 충돌! 페널티: -50")
    
    # 5. 너무 오래 살아있으면 작은 페널티 (무한 루프 방지)
    if steps > 500:
        reward -= 0.5
    
    return reward

def visualize_training_progress(scores, avg_scores, save_path="training_progress.png"):
    """훈련 진행 상황 시각화"""
    plt.style.use('dark_background')  # 다크모드
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 개별 점수
    ax1.plot(scores, color='#64ffda', alpha=0.6, linewidth=1)
    ax1.set_title('🎯 에피소드별 점수', color='#e0e6ed', fontsize=14, fontweight='bold')
    ax1.set_xlabel('에피소드', color='#e0e6ed')
    ax1.set_ylabel('점수', color='#e0e6ed')
    ax1.grid(True, alpha=0.3)
    
    # 이동 평균
    if avg_scores:
        ax2.plot(avg_scores, color='#bb86fc', linewidth=3)
        ax2.set_title('📈 이동 평균 점수 (100 에피소드)', color='#e0e6ed', fontsize=14, fontweight='bold')
        ax2.set_xlabel('에피소드 (x100)', color='#e0e6ed') 
        ax2.set_ylabel('평균 점수', color='#e0e6ed')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, facecolor='#1a1a1a', dpi=150)
    plt.show()
    
    print(f"{Colors.GREEN}📊 훈련 그래프 저장: {save_path}{Colors.RESET}")

def run_improved_training(episodes=3000, save_interval=500, model_name="snake_improved_agent"):
    """개선된 훈련 메인 함수"""
    
    print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}🎯 개선된 Snake AI 훈련 시작{Colors.RESET}")
    print(f"{Colors.WHITE}목표: 평균 점수 0.7 → 10+ 달성{Colors.RESET}")
    print(f"{Colors.BLUE}에피소드: {episodes}, 저장 간격: {save_interval}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
    
    # 환경 및 에이전트 초기화
    game = SnakeGameAI()
    agent = ImprovedQLearningAgent()
    
    # 훈련 기록
    scores = []
    avg_scores = []
    best_score = 0
    best_avg_score = 0
    
    start_time = time.time()
    
    for episode in range(episodes):
        # 에피소드 초기화
        state = game.reset()
        prev_score = 0
        total_reward = 0
        steps = 0
        
        while True:
            # 행동 선택 및 실행
            action = agent.get_action(state)
            
            # 게임 스텝 (실제 반환값에 맞춤)
            play_result = game.play_step(action)
            curr_score = game.score if hasattr(game, 'score') else 0
            
            # 게임 종료 조건
            done = game.is_collision() if hasattr(game, 'is_collision') else False
            
            # 개선된 보상 계산
            reward = improved_reward_function(game, prev_score, curr_score, done, steps)
            total_reward += reward
            
            # 다음 상태
            next_state = game.get_state()
            
            # Q-러닝 업데이트
            agent.update_q_table(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            prev_score = curr_score
            steps += 1
            
            # 종료 조건
            if done or steps > 1000:  # 최대 스텝 증가
                break
        
        # 에피소드 결과 기록
        final_score = curr_score
        scores.append(final_score)
        
        # 최고 점수 업데이트
        if final_score > best_score:
            best_score = final_score
        
        # 이동 평균 계산
        if len(scores) >= 100:
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            if avg_score > best_avg_score:
                best_avg_score = avg_score
        
        # Epsilon 감소
        agent.decay_epsilon()
        
        # 진행 상황 출력
        if episode % 100 == 0:
            elapsed_time = time.time() - start_time
            recent_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            
            print(f"{Colors.YELLOW}에피소드 {episode:4d}{Colors.RESET} | "
                  f"점수: {final_score:2d} | "
                  f"평균: {recent_avg:5.2f} | "
                  f"최고: {best_score:2d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"시간: {elapsed_time/60:.1f}분")
            
            # 조기 성공 조건
            if recent_avg >= 15:
                print(f"{Colors.GREEN}🎉 목표 달성! 평균 점수 15점 돌파{Colors.RESET}")
                break
        
        # 중간 저장
        if episode % save_interval == 0 and episode > 0:
            intermediate_name = f"{model_name}_ep{episode}.pkl"
            save_model(agent, scores, avg_scores, intermediate_name, episode)
            print(f"{Colors.BLUE}💾 중간 저장: {intermediate_name}{Colors.RESET}")
    
    # 최종 결과
    final_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    total_time = time.time() - start_time
    
    print(f"\n{Colors.GREEN}🎉 훈련 완료!{Colors.RESET}")
    print(f"{Colors.WHITE}{'='*50}{Colors.RESET}")
    print(f"{Colors.CYAN}📊 최종 통계:{Colors.RESET}")
    print(f"   최종 평균 점수: {final_avg:.2f}")
    print(f"   최고 점수: {best_score}")
    print(f"   최고 평균 점수: {best_avg_score:.2f}")
    print(f"   총 훈련 시간: {total_time/60:.1f}분")
    print(f"   Q-테이블 크기: {len(agent.q_table)}")
    
    # 성능 평가
    if final_avg >= 15:
        print(f"{Colors.GREEN}🌟 탁월한 성능! 목표 대폭 초과 달성{Colors.RESET}")
    elif final_avg >= 10:
        print(f"{Colors.GREEN}🎯 목표 달성! 우수한 성능{Colors.RESET}")
    elif final_avg >= 5:
        print(f"{Colors.YELLOW}👍 큰 개선! 추가 훈련 권장{Colors.RESET}")
    else:
        print(f"{Colors.RED}🔄 추가 개선 필요{Colors.RESET}")
    
    # 최종 모델 저장
    final_name = f"{model_name}_final.pkl"
    save_model(agent, scores, avg_scores, final_name, episodes)
    
    # 시각화
    visualize_training_progress(scores, avg_scores)
    
    return {
        'agent': agent,
        'scores': scores,
        'avg_scores': avg_scores,
        'final_avg': final_avg,
        'best_score': best_score,
        'training_time': total_time
    }

def save_model(agent, scores, avg_scores, filename, episodes):
    """모델 저장"""
    model_data = {
        'q_table': agent.q_table,
        'epsilon': agent.epsilon,
        'learning_rate': agent.learning_rate,
        'discount_factor': agent.discount_factor,
        'epsilon_min': agent.epsilon_min,
        'epsilon_decay': agent.epsilon_decay,
        'scores': scores,
        'avg_scores': avg_scores,
        'episodes': episodes,
        'timestamp': time.time()
    }
    
    # 프로젝트 루트에 저장
    save_path = os.path.join(project_root, filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"{Colors.GREEN}✅ 모델 저장 완료: {save_path}{Colors.RESET}")

def compare_with_previous():
    """기존 모델과 성능 비교"""
    print(f"\n{Colors.CYAN}📊 기존 모델과 성능 비교{Colors.RESET}")
    
    # 기존 모델 파일들 찾기
    old_models = []
    for file in os.listdir(project_root):
        if file.endswith('.pkl') and 'improved' not in file:
            old_models.append(file)
    
    if old_models:
        print(f"{Colors.YELLOW}발견된 기존 모델:{Colors.RESET}")
        for model in old_models:
            print(f"   📦 {model}")
        
        # 첫 번째 모델 분석
        try:
            with open(os.path.join(project_root, old_models[0]), 'rb') as f:
                old_data = pickle.load(f)
            
            if 'scores' in old_data:
                old_avg = np.mean(old_data['scores'][-100:]) if len(old_data['scores']) >= 100 else np.mean(old_data['scores'])
                print(f"{Colors.WHITE}기존 평균 점수: {old_avg:.2f}{Colors.RESET}")
                print(f"{Colors.WHITE}개선 목표: {old_avg:.2f} → 10+ (약 {1000/old_avg:.0f}% 향상){Colors.RESET}")
            
        except Exception as e:
            print(f"{Colors.RED}기존 모델 분석 실패: {e}{Colors.RESET}")

def main():
    """메인 실행 함수"""
    print(f"{Colors.MAGENTA}🚀 개선된 Snake AI 훈련 시작{Colors.RESET}")
    
    # 기존 모델과 비교
    compare_with_previous()
    
    # 개선된 훈련 실행
    result = run_improved_training(
        episodes=3000,      # 충분한 훈련
        save_interval=500,  # 정기 저장
        model_name="snake_improved_agent"
    )
    
    print(f"\n{Colors.CYAN}💡 다음 단계:{Colors.RESET}")
    print(f"{Colors.WHITE}1. demo_runner.py로 개선된 성능 확인{Colors.RESET}")
    print(f"{Colors.WHITE}2. 만족스럽지 않다면 더 긴 훈련 (5000+ 에피소드){Colors.RESET}")
    print(f"{Colors.WHITE}3. DQN 구현 고려{Colors.RESET}")

if __name__ == "__main__":
    main()