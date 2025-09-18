# 🚀 뱀게임 AI - 학습 스크립트 (train.py)
# ================================================================
# 작성자: AI/빅데이터 석사과정생
# 환경: VSCode/Colab (다크모드 최적화)
# 목표: Q-Learning으로 뱀게임을 마스터하는 AI 훈련
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime
from tqdm import tqdm
import json

# 체계적인 구조에서 패키지 import
from .game import SnakeGameAI
from .agent import QLearningAgent

# 시각화는 utils에서 import
from utils.visualization import TrainingVisualizer

# 유틸리티 함수들 import
from utils import create_directory, get_timestamp, save_json, setup_logging

# Colab 환경에서 실행 시 주석 해제
# from IPython.display import clear_output

class SnakeAITrainer:
    """
    뱀게임 AI 학습을 관리하는 메인 클래스
    
    주요 기능:
    - 학습 과정 전체 관리
    - 실시간 진행 상황 모니터링
    - 모델 저장/로드 관리
    - 성능 평가 및 시각화
    - 하이퍼파라미터 튜닝 지원
    
    Attributes:
        game: 뱀게임 환경 인스턴스
        agent: Q-Learning 에이전트 인스턴스  
        visualizer: 학습 시각화 도구
        config: 학습 설정 딕셔너리
        training_stats: 학습 통계 정보
    """
    
    def __init__(self, config=None):
        """
        AI 트레이너 초기화
        
        Args:
            config (dict, optional): 학습 설정 딕셔너리
            
        설명:
            - 게임 환경, 에이전트, 시각화 도구 초기화
            - 기본 설정값 적용 또는 사용자 설정 로드
            - 로그 디렉토리 생성
        """
        # 기본 설정값
        self.default_config = {
            'episodes': 5000,           # 총 학습 에피소드 수
            'max_steps': 1000,          # 에피소드당 최대 스텝
            'display_game': True,       # 게임 화면 표시 여부
            'display_interval': 100,    # 화면 표시 간격
            'save_interval': 500,       # 모델 저장 간격
            'eval_interval': 100,       # 평가 간격
            'log_interval': 50,         # 로그 출력 간격
            'target_score': 20,         # 목표 점수
            'early_stopping': True,     # 조기 종료 사용 여부
            'patience': 1000,           # 조기 종료 대기 에피소드
            
            # Q-Learning 하이퍼파라미터
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            
            # 게임 설정
            'game_width': 640,
            'game_height': 480,
            'game_speed': 20
        }
        
        # 설정 적용
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 학습 통계 초기화
        self.training_stats = {
            'start_time': None,
            'end_time': None,
            'total_episodes': 0,
            'best_score': 0,
            'best_episode': 0,
            'avg_score_last_100': 0,
            'convergence_episode': None,
            'training_time': 0
        }
        
        # 로그 디렉토리 생성
        self.log_dir = f"logs/snake_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        print("🚀 뱀게임 AI 트레이너가 초기화되었습니다!")
        print(f"📁 로그 디렉토리: {self.log_dir}")
        self._print_config()
    
    def _initialize_components(self):
        """
        게임, 에이전트, 시각화 도구를 초기화하는 내부 함수
        
        Returns:
            None
            
        설명:
            - 뱀게임 환경 생성
            - Q-Learning 에이전트 생성
            - 시각화 도구 초기화
        """
        # 게임 환경 초기화
        self.game = SnakeGameAI(
            w=self.config['game_width'],
            h=self.config['game_height'],
            display=self.config['display_game']
        )
        
        # Q-Learning 에이전트 초기화
        self.agent = QLearningAgent(
            lr=self.config['learning_rate'],
            gamma=self.config['discount_factor'],
            epsilon=self.config['epsilon'],
            epsilon_min=self.config['epsilon_min'],
            epsilon_decay=self.config['epsilon_decay']
        )
        
        # 시각화 도구 초기화
        self.visualizer = TrainingVisualizer()
    
    def _print_config(self):
        """
        현재 설정을 출력하는 내부 함수
        
        Returns:
            None
        """
        print("\n" + "="*60)
        print("⚙️  학습 설정")
        print("="*60)
        print(f"📊 총 에피소드: {self.config['episodes']:,}")
        print(f"🎯 목표 점수: {self.config['target_score']}")
        print(f"🧠 학습률: {self.config['learning_rate']}")
        print(f"🎲 초기 탐험율: {self.config['epsilon']}")
        print(f"💾 저장 간격: {self.config['save_interval']} 에피소드")
        print(f"📺 화면 표시: {'ON' if self.config['display_game'] else 'OFF'}")
        print("="*60)
    
    def train(self, resume_from=None):
        """
        AI 학습을 실행하는 메인 함수
        
        Args:
            resume_from (str, optional): 재개할 모델 파일 경로
            
        Returns:
            dict: 학습 결과 통계
            
        설명:
            1. 모델 로드 (재개하는 경우)
            2. 에피소드별 학습 루프 실행
            3. 실시간 진행 상황 모니터링
            4. 주기적 모델 저장 및 평가
            5. 조기 종료 조건 확인
            6. 최종 결과 저장 및 시각화
        """
        print("\n🎓 AI 학습을 시작합니다!")
        
        # 시작 시간 기록
        self.training_stats['start_time'] = datetime.now()
        start_episode = 0
        
        # 모델 재개 (선택사항)
        if resume_from and os.path.exists(resume_from):
            if self.agent.load_model(resume_from):
                start_episode = len(self.agent.training_history['scores'])
                print(f"🔄 {start_episode} 에피소드부터 학습을 재개합니다.")
        
        # 조기 종료를 위한 변수들
        best_avg_score = -float('inf')
        no_improvement_count = 0
        
        try:
            # 학습 루프
            with tqdm(range(start_episode, self.config['episodes']), 
                     desc="🧠 AI 학습 중", 
                     ncols=100, 
                     bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
                
                for episode in pbar:
                    # 한 에피소드 실행
                    score, total_reward = self._run_episode(episode)
                    
                    # 학습 기록 업데이트
                    self.agent.update_training_history(score, total_reward, episode)
                    self.training_stats['total_episodes'] = episode + 1
                    
                    # 최고 점수 업데이트
                    if score > self.training_stats['best_score']:
                        self.training_stats['best_score'] = score
                        self.training_stats['best_episode'] = episode
                    
                    # 진행 상황 업데이트
                    self._update_progress_bar(pbar, episode, score, total_reward)
                    
                    # 주기적 로그 출력
                    if (episode + 1) % self.config['log_interval'] == 0:
                        self._print_progress(episode)
                    
                    # 주기적 모델 저장
                    if (episode + 1) % self.config['save_interval'] == 0:
                        self._save_checkpoint(episode)
                    
                    # 주기적 평가
                    if (episode + 1) % self.config['eval_interval'] == 0:
                        avg_score = self._evaluate_performance(episode)
                        
                        # 조기 종료 확인
                        if self.config['early_stopping']:
                            if avg_score > best_avg_score:
                                best_avg_score = avg_score
                                no_improvement_count = 0
                            else:
                                no_improvement_count += self.config['eval_interval']
                            
                            # 목표 달성 또는 수렴 확인
                            if (avg_score >= self.config['target_score'] or 
                                no_improvement_count >= self.config['patience']):
                                self._handle_early_stopping(episode, avg_score)
                                break
        
        except KeyboardInterrupt:
            print("\n⚠️ 사용자에 의해 학습이 중단되었습니다.")
        
        finally:
            # 학습 종료 처리
            self._finalize_training()
        
        return self.training_stats
    
    def _run_episode(self, episode):
        """
        한 에피소드를 실행하는 내부 함수
        
        Args:
            episode (int): 현재 에피소드 번호
            
        Returns:
            tuple: (최종 점수, 총 보상)
            
        설명:
            1. 게임 환경 리셋
            2. 스텝별 게임 진행
            3. 상태-행동-보상 기록
            4. Q-Learning 업데이트
        """
        # 게임 초기화
        self.game.reset()
        state = self.game.get_state()
        total_reward = 0
        
        # 화면 표시 여부 결정
        display_this_episode = (episode % self.config['display_interval'] == 0)
        
        for step in range(self.config['max_steps']):
            # 행동 선택
            action = self.agent.get_action(state)
            
            # 행동 수행
            reward, done, score = self.game.play_step(action)
            next_state = self.game.get_state()
            total_reward += reward
            
            # 경험 저장
            self.agent.remember(state, action, reward, next_state, done)
            
            # Q-Learning 업데이트
            self.agent.learn(state, action, reward, next_state, done)
            
            # 상태 업데이트
            state = next_state
            
            # 게임 종료 확인
            if done:
                break
            
            # 화면 표시 속도 조절
            if display_this_episode and self.config['display_game']:
                time.sleep(0.05)  # 관찰하기 쉽도록 속도 조절
        
        return score, total_reward
    
    def _update_progress_bar(self, pbar, episode, score, total_reward):
        """
        진행률 표시줄을 업데이트하는 내부 함수
        
        Args:
            pbar: tqdm 진행률 표시줄 객체
            episode (int): 현재 에피소드
            score (int): 에피소드 점수
            total_reward (float): 에피소드 총 보상
            
        Returns:
            None
        """
        # 최근 100 에피소드 평균 계산
        recent_scores = self.agent.training_history['scores'][-100:]
        avg_score = np.mean(recent_scores) if recent_scores else 0
        
        # 진행률 표시줄 업데이트
        pbar.set_postfix({
            '점수': f'{score:3d}',
            '평균': f'{avg_score:5.1f}',
            '최고': f'{self.training_stats["best_score"]:3d}',
            'ε': f'{self.agent.epsilon:.3f}'
        })
    
    def _print_progress(self, episode):
        """
        상세한 진행 상황을 출력하는 내부 함수
        
        Args:
            episode (int): 현재 에피소드 번호
            
        Returns:
            None
        """
        recent_scores = self.agent.training_history['scores'][-self.config['log_interval']:]
        recent_rewards = self.agent.training_history['rewards'][-self.config['log_interval']:]
        
        if recent_scores:
            avg_score = np.mean(recent_scores)
            avg_reward = np.mean(recent_rewards)
            q_stats = self.agent.get_q_table_stats()
            
            print(f"\n📊 에피소드 {episode+1:,}")
            print(f"   점수: {recent_scores[-1]:3d} | 평균: {avg_score:5.1f} | 최고: {self.training_stats['best_score']:3d}")
            print(f"   보상: {avg_reward:7.2f} | Epsilon: {self.agent.epsilon:.4f}")
            print(f"   Q-테이블: {q_stats['size']:,}개 상태 | 평균 Q값: {q_stats['avg_q_value']:.3f}")
    
    def _evaluate_performance(self, episode):
        """
        현재 성능을 평가하는 내부 함수
        
        Args:
            episode (int): 현재 에피소드 번호
            
        Returns:
            float: 최근 100 에피소드 평균 점수
        """
        recent_scores = self.agent.training_history['scores'][-100:]
        avg_score = np.mean(recent_scores) if len(recent_scores) >= 50 else 0
        
        self.training_stats['avg_score_last_100'] = avg_score
        
        print(f"\n🎯 성능 평가 (에피소드 {episode+1})")
        print(f"   최근 100 에피소드 평균: {avg_score:.2f}")
        print(f"   목표 점수: {self.config['target_score']}")
        print(f"   달성률: {(avg_score / self.config['target_score'] * 100):5.1f}%")
        
        return avg_score
    
    def _handle_early_stopping(self, episode, avg_score):
        """
        조기 종료 조건을 처리하는 내부 함수
        
        Args:
            episode (int): 현재 에피소드 번호
            avg_score (float): 평균 점수
            
        Returns:
            None
        """
        self.training_stats['convergence_episode'] = episode + 1
        
        if avg_score >= self.config['target_score']:
            print(f"\n🎉 목표 달성! 에피소드 {episode+1}에서 평균 점수 {avg_score:.2f} 달성!")
        else:
            print(f"\n⏹️ 학습 수렴! 에피소드 {episode+1}에서 조기 종료")
        
        # 최종 모델 저장
        final_model_path = f"models/snake_ai_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        self.agent.save_model(final_model_path)
    
    def _save_checkpoint(self, episode):
        """
        중간 체크포인트를 저장하는 내부 함수
        
        Args:
            episode (int): 현재 에피소드 번호
            
        Returns:
            None
        """
        checkpoint_path = f"models/snake_ai_checkpoint_ep{episode+1}.pkl"
        self.agent.save_model(checkpoint_path)
        
        # 설정 파일도 함께 저장
        config_path = f"{self.log_dir}/config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _finalize_training(self):
        """
        학습 종료 후 정리 작업을 수행하는 내부 함수
        
        Returns:
            None
        """
        # 종료 시간 기록
        self.training_stats['end_time'] = datetime.now()
        self.training_stats['training_time'] = (
            self.training_stats['end_time'] - self.training_stats['start_time']
        ).total_seconds() / 60  # 분 단위
        
        # 최종 모델 저장
        if not self.training_stats['convergence_episode']:
            final_model_path = f"models/snake_ai_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            self.agent.save_model(final_model_path)
        
        # 학습 통계 저장
        stats_path = f"{self.log_dir}/training_stats.json"
        with open(stats_path, 'w') as f:
            # datetime 객체를 문자열로 변환
            stats_to_save = self.training_stats.copy()
            if stats_to_save['start_time']:
                stats_to_save['start_time'] = stats_to_save['start_time'].isoformat()
            if stats_to_save['end_time']:
                stats_to_save['end_time'] = stats_to_save['end_time'].isoformat()
            json.dump(stats_to_save, f, indent=2)
        
        # 최종 결과 출력
        self._print_final_results()
        
        # 학습 그래프 생성
        graph_path = f"{self.log_dir}/training_progress.png"
        self.visualizer.plot_training_progress(self.agent, save_path=graph_path)
    
    def _print_final_results(self):
        """
        최종 학습 결과를 출력하는 내부 함수
        
        Returns:
            None
        """
        print("\n" + "="*60)
        print("🏆 학습 완료!")
        print("="*60)
        print(f"⏱️  총 학습 시간: {self.training_stats['training_time']:.1f}분")
        print(f"📊 총 에피소드: {self.training_stats['total_episodes']:,}")
        print(f"🥇 최고 점수: {self.training_stats['best_score']} (에피소드 {self.training_stats['best_episode']+1})")
        print(f"📈 최종 평균 점수: {self.training_stats['avg_score_last_100']:.2f}")
        
        if self.training_stats['convergence_episode']:
            print(f"🎯 수렴 에피소드: {self.training_stats['convergence_episode']}")
        
        q_stats = self.agent.get_q_table_stats()
        print(f"🧠 Q-테이블 크기: {q_stats['size']:,}개 상태")
        print(f"📁 로그 디렉토리: {self.log_dir}")
        print("="*60)
    
    def test_trained_agent(self, model_path, test_episodes=10):
        """
        학습된 에이전트를 테스트하는 함수
        
        Args:
            model_path (str): 테스트할 모델 파일 경로
            test_episodes (int): 테스트 에피소드 수
            
        Returns:
            dict: 테스트 결과 통계
        """
        print(f"\n🧪 학습된 AI 테스트 (모델: {model_path})")
        
        # 모델 로드
        if not self.agent.load_model(model_path):
            print("❌ 모델 로드 실패")
            return None
        
        # 탐험 비활성화 (테스트 모드)
        original_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0  # 완전히 학습된 정책 사용
        
        test_scores = []
        test_rewards = []
        
        print(f"🎮 {test_episodes}번의 테스트 게임을 실행합니다...")
        
        for episode in range(test_episodes):
            # 게임 실행
            self.game.reset()
            state = self.game.get_state()
            total_reward = 0
            
            for step in range(self.config['max_steps']):
                action, q_values = self.agent.get_best_action_for_state(state)
                reward, done, score = self.game.play_step(action)
                state = self.game.get_state()
                total_reward += reward
                
                if done:
                    break
            
            test_scores.append(score)
            test_rewards.append(total_reward)
            
            print(f"  게임 {episode+1}: 점수 {score}, 보상 {total_reward:.1f}")
        
        # 원래 epsilon 복원
        self.agent.epsilon = original_epsilon
        
        # 테스트 결과 분석
        test_results = {
            'avg_score': np.mean(test_scores),
            'max_score': np.max(test_scores),
            'min_score': np.min(test_scores),
            'std_score': np.std(test_scores),
            'avg_reward': np.mean(test_rewards),
            'scores': test_scores,
            'rewards': test_rewards
        }
        
        print(f"\n📊 테스트 결과:")
        print(f"   평균 점수: {test_results['avg_score']:.2f} ± {test_results['std_score']:.2f}")
        print(f"   최고 점수: {test_results['max_score']}")
        print(f"   최저 점수: {test_results['min_score']}")
        print(f"   평균 보상: {test_results['avg_reward']:.2f}")
        
        return test_results

# ================================================================
# 실행 함수들
# ================================================================

def main_training():
    """
    메인 학습 실행 함수
    
    Returns:
        None
        
    설명:
        - 기본 설정으로 AI 학습 시작
        - 사용자가 쉽게 실행할 수 있는 래퍼 함수
    """
    print("🎮 뱀게임 AI 학습을 시작합니다!")
    
    # 기본 설정
    config = {
        'episodes': 3000,
        'target_score': 15,
        'display_game': True,
        'display_interval': 50,
        'save_interval': 300,
        'learning_rate': 0.1,
        'epsilon_decay': 0.996
    }
    
    # 트레이너 생성 및 학습 시작
    trainer = SnakeAITrainer(config)
    results = trainer.train()
    
    print("🎉 학습이 완료되었습니다!")
    return results

def quick_demo():
    """
    빠른 데모를 위한 함수 (짧은 학습)
    
    Returns:
        None
    """
    print("⚡ 빠른 데모 모드 (500 에피소드)")
    
    config = {
        'episodes': 500,
        'target_score': 10,
        'display_game': True,
        'display_interval': 25,
        'save_interval': 100,
        'log_interval': 25
    }
    
    trainer = SnakeAITrainer(config)
    results = trainer.train()
    
    return results

def advanced_training():
    """
    고급 설정으로 장시간 학습하는 함수
    
    Returns:
        None
    """
    print("🚀 고급 모드: 장시간 정밀 학습")
    
    config = {
        'episodes': 10000,
        'target_score': 25,
        'display_game': False,  # 학습 속도 향상
        'display_interval': 200,
        'save_interval': 500,
        'learning_rate': 0.05,  # 더 안정적인 학습
        'epsilon_decay': 0.9995,  # 더 천천히 탐험 감소
        'patience': 2000  # 더 오래 기다림
    }
    
    trainer = SnakeAITrainer(config)
    results = trainer.train()
    
    return results

if __name__ == "__main__":
    print("🐍 뱀게임 AI 메인 스크립트가 준비되었습니다!")
    print("\n실행 옵션:")
    print("1. main_training() - 기본 학습 (3000 에피소드)")
    print("2. quick_demo() - 빠른 데모 (500 에피소드)")  
    print("3. advanced_training() - 고급 학습 (10000 에피소드)")
    print("\n예시:")
    print(">>> results = main_training()")
    
    results = main_training()