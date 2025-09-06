#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_runner.py - 학습된 Q-Learning Snake AI 데모 실행기

훈련 완료된 에이전트의 실제 게임 플레이를 시각화하여 보여줍니다.
로컬 환경과 Google Colab 자ㄱ동.
"""

import os
import sys
import time
import pickle
import numpy as np
from typing import Dict, Any, Tuple, Optional

# 환경 감지
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

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

def print_demo_banner():
    """데모 시작 배너 출력"""
    print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}🎮 Q-Learning Snake AI 데모 실행기{Colors.RESET}")
    print(f"{Colors.WHITE}🤖 훈련된 AI 에이전트가 스네이크 게임을 플레이합니다{Colors.RESET}")
    env_text = "Google Colab" if IN_COLAB else "로컬 환경"
    print(f"{Colors.BLUE}📱 실행 환경: {env_text}{Colors.RESET}")
    print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")

class QLearningAgent:
    """Q-Learning 에이전트"""
    
    def __init__(self, q_table, epsilon=0.01):
        self.q_table = q_table
        self.epsilon = epsilon
        print(f"{Colors.GREEN}🤖 에이전트 로드 완료 - Q테이블: {len(q_table)} 상태{Colors.RESET}")
    
    def act(self, state):
        """행동 선택"""
        state_key = str(state)
        
        if state_key not in self.q_table:
            return np.random.randint(0, 3)
        
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 3)
        else:
            q_values = self.q_table[state_key]
            return np.argmax(q_values)

class GameEnvironment:
    """게임 환경 래퍼"""
    
    def __init__(self):
        self.game = None
        self.last_score = 0
        self.steps = 0
        self._load_game()
    
    def _load_game(self):
        """게임 로드"""
        try:
            from games.snake.game import SnakeGameAI
            self.game = SnakeGameAI()
            print(f"{Colors.GREEN}🎮 게임 환경 로드 완료{Colors.RESET}")
        except ImportError as e:
            print(f"{Colors.RED}❌ 게임 로드 실패: {e}{Colors.RESET}")
            raise
    
    def reset(self):
        """게임 초기화"""
        if self.game is None:
            raise RuntimeError("게임이 로드되지 않았습니다")
        
        self.game.reset()
        self.last_score = 0
        self.steps = 0
        state = self.game.get_state()
        return state
    
    def step(self, action):
        """한 스텝 실행"""
        if self.game is None:
            raise RuntimeError("게임이 로드되지 않았습니다")
        
        self.steps += 1
        
        # play_step 호출
        result = self.game.play_step(action)
        
        # 반환값 처리
        if isinstance(result, (int, float)):
            score = int(result)
        elif isinstance(result, tuple) and len(result) >= 1:
            score = int(result[0])
        else:
            score = 0
        
        reward = score - self.last_score
        done = False
        
        # 현재 상태 얻기
        state = self.game.get_state()
        
        # 충돌 검사
        try:
            if hasattr(self.game, 'is_collision'):
                if self.game.is_collision():
                    done = True
                    reward = -10
        except Exception:
            pass
        
        # 최대 스텝 제한
        if self.steps > 500:
            done = True
        
        self.last_score = score
        
        info = {
            'score': score,
            'steps': self.steps
        }
        
        return state, reward, done, info

def find_model_file():
    """모델 파일 찾기"""
    possible_files = [
        "snake_ai_agent_final.pkl",
        "snake_q_agent.pkl", 
        "q_agent.pkl",
        "agent.pkl"
    ]
    
    print(f"{Colors.YELLOW}🔍 모델 파일 검색 중...{Colors.RESET}")
    
    # 직접 확인
    for filename in possible_files:
        if os.path.exists(filename):
            print(f"{Colors.GREEN}✅ 발견: {filename}{Colors.RESET}")
            return filename
    
    # .pkl 파일 자동 검색
    pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if pkl_files:
        print(f"{Colors.GREEN}✅ 자동 발견: {pkl_files[0]}{Colors.RESET}")
        return pkl_files[0]
    
    print(f"{Colors.RED}❌ 모델 파일을 찾을 수 없습니다{Colors.RESET}")
    return None

def load_agent():
    """에이전트 로드"""
    model_file = find_model_file()
    if not model_file:
        return None
    
    print(f"{Colors.CYAN}📦 모델 로드 중: {model_file}{Colors.RESET}")
    
    try:
        with open(model_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'q_table' in data:
            q_table = data['q_table']
            epsilon = data.get('epsilon', 0.01)
            
            print(f"{Colors.BLUE}📊 Q-테이블 크기: {len(q_table)}{Colors.RESET}")
            print(f"{Colors.BLUE}🎯 Epsilon: {epsilon:.4f}{Colors.RESET}")
            
            return QLearningAgent(q_table, epsilon)
        else:
            print(f"{Colors.RED}❌ Q-테이블을 찾을 수 없습니다{Colors.RESET}")
            return None
            
    except Exception as e:
        print(f"{Colors.RED}❌ 로드 실패: {e}{Colors.RESET}")
        return None

def run_episode(game_env, agent, episode_num, max_steps=300):
    """단일 에피소드 실행"""
    print(f"\n{Colors.CYAN}🎮 에피소드 {episode_num} 시작...{Colors.RESET}")
    
    try:
        state = game_env.reset()
        steps = 0
        done = False
        
        while not done and steps < max_steps:
            action = agent.act(state)
            next_state, reward, done, info = game_env.step(action)
            
            steps += 1
            
            # 진행 상황 출력
            if steps % 50 == 0:
                print(f"   📊 {steps} 스텝: 점수 {info['score']}")
            
            state = next_state
        
        final_score = info['score']
        print(f"   🏆 에피소드 {episode_num} 완료!")
        print(f"      📊 최종 점수: {final_score}")
        print(f"      ⏱️ 생존 시간: {steps} 스텝")
        
        return {
            'episode': episode_num,
            'score': final_score,
            'steps': steps,
            'success': True
        }
        
    except Exception as e:
        print(f"   {Colors.RED}❌ 에피소드 {episode_num} 실패: {e}{Colors.RESET}")
        return {
            'episode': episode_num,
            'score': 0,
            'steps': 0,
            'success': False
        }

def run_multiple_episodes(game_env, agent, num_episodes=5, max_steps=300):
    """여러 에피소드 실행"""
    print(f"\n{Colors.MAGENTA}🚀 {num_episodes}개 에피소드 데모 시작{Colors.RESET}")
    
    results = []
    
    for episode in range(1, num_episodes + 1):
        result = run_episode(game_env, agent, episode, max_steps)
        results.append(result)
        
        # 에피소드 간 대기
        if episode < num_episodes:
            print(f"{Colors.YELLOW}⏳ 다음 에피소드까지 1초 대기...{Colors.RESET}")
            time.sleep(1)
    
    # 성공한 에피소드들의 통계
    successful = [r for r in results if r['success']]
    
    if successful:
        scores = [r['score'] for r in successful]
        steps = [r['steps'] for r in successful]
        
        print(f"\n{Colors.MAGENTA}📊 데모 완료 - 전체 통계{Colors.RESET}")
        print(f"{Colors.WHITE}{'='*50}{Colors.RESET}")
        print(f"{Colors.CYAN}🎯 평균 점수: {Colors.WHITE}{sum(scores)/len(scores):.1f}{Colors.RESET}")
        print(f"{Colors.GREEN}🏆 최고 점수: {Colors.WHITE}{max(scores)}{Colors.RESET}")
        print(f"{Colors.BLUE}⏱️ 평균 생존: {Colors.WHITE}{sum(steps)/len(steps):.1f} 스텝{Colors.RESET}")
        print(f"{Colors.BLUE}⏱️ 최장 생존: {Colors.WHITE}{max(steps)} 스텝{Colors.RESET}")
        print(f"{Colors.YELLOW}✅ 성공률: {Colors.WHITE}{len(successful)}/{num_episodes}{Colors.RESET}")
        print(f"{Colors.WHITE}{'='*50}{Colors.RESET}")
        
        # 성능 평가
        avg_score = sum(scores) / len(scores)
        if avg_score >= 15:
            print(f"\n{Colors.GREEN}🌟 훌륭한 성능! AI가 매우 잘 학습되었습니다.{Colors.RESET}")
        elif avg_score >= 8:
            print(f"\n{Colors.YELLOW}👍 괜찮은 성능! 더 개선할 여지가 있습니다.{Colors.RESET}")
        else:
            print(f"\n{Colors.YELLOW}🔄 기본 성능! 더 많은 훈련이 도움될 것입니다.{Colors.RESET}")
        
        return {
            'avg_score': avg_score,
            'max_score': max(scores),
            'success_rate': len(successful) / num_episodes,
            'results': results
        }
    else:
        print(f"\n{Colors.RED}❌ 모든 에피소드가 실패했습니다{Colors.RESET}")
        return {'success_rate': 0, 'results': results}

def get_user_settings():
    """사용자 설정 입력"""
    if IN_COLAB:
        # Colab에서는 기본값 사용
        print(f"\n{Colors.CYAN}📱 Colab 환경 - 자동 설정{Colors.RESET}")
        num_episodes = 3
        max_steps = 200
    else:
        # 로컬에서는 사용자 입력
        print(f"\n{Colors.YELLOW}❓ 설정을 입력하세요{Colors.RESET}")
        
        try:
            episodes_input = input(f"{Colors.CYAN}에피소드 수 (기본값 5): {Colors.RESET}").strip()
            num_episodes = int(episodes_input) if episodes_input else 5
            num_episodes = max(1, min(num_episodes, 10))
        except ValueError:
            num_episodes = 5
        
        max_steps = 300
    
    print(f"{Colors.WHITE}   • 에피소드 수: {num_episodes}{Colors.RESET}")
    print(f"{Colors.WHITE}   • 최대 스텝: {max_steps}{Colors.RESET}")
    
    return num_episodes, max_steps

def main():
    """메인 함수"""
    print_demo_banner()
    
    # 현재 위치 출력
    print(f"\n{Colors.BLUE}📍 현재 작업 디렉토리: {os.getcwd()}{Colors.RESET}")
    
    # 1. 에이전트 로드
    agent = load_agent()
    if not agent:
        print(f"\n{Colors.RED}🚫 에이전트를 로드할 수 없습니다{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 해결 방법:{Colors.RESET}")
        print(f"{Colors.WHITE}   1. 먼저 훈련을 실행하세요: python main.py{Colors.RESET}")
        print(f"{Colors.WHITE}   2. .pkl 파일이 현재 폴더에 있는지 확인하세요{Colors.RESET}")
        return
    
    # 2. 게임 환경 준비
    try:
        game_env = GameEnvironment()
    except Exception as e:
        print(f"\n{Colors.RED}❌ 게임 환경 초기화 실패: {e}{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 해결 방법:{Colors.RESET}")
        print(f"{Colors.WHITE}   1. games/snake/ 폴더 확인{Colors.RESET}")
        print(f"{Colors.WHITE}   2. __init__.py 파일들 확인{Colors.RESET}")
        return
    
    # 3. 사용자 설정
    num_episodes, max_steps = get_user_settings()
    
    # 4. 데모 실행
    try:
        stats = run_multiple_episodes(game_env, agent, num_episodes, max_steps)
        
        # 5. 다음 단계 제안
        if stats.get('success_rate', 0) > 0:
            print(f"\n{Colors.CYAN}💡 다음 단계 제안:{Colors.RESET}")
            print(f"{Colors.WHITE}   1️⃣ 더 긴 에피소드로 성능 확인{Colors.RESET}")
            print(f"{Colors.WHITE}   2️⃣ 시각화 그래프 생성{Colors.RESET}")
            print(f"{Colors.WHITE}   3️⃣ 하이퍼파라미터 최적화{Colors.RESET}")
            print(f"{Colors.WHITE}   4️⃣ DQN 구현으로 업그레이드{Colors.RESET}")
        
    except Exception as e:
        print(f"\n{Colors.RED}❌ 데모 실행 중 오류: {e}{Colors.RESET}")
        import traceback
        print(f"{Colors.YELLOW}🔍 상세 오류:{Colors.RESET}")
        traceback.print_exc()
    
    print(f"\n{Colors.GREEN}🎉 데모 완료! 수고하셨습니다.{Colors.RESET}")

if __name__ == "__main__":
    main()