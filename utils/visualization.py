# 📊 utils/visualization.py
# ================================================================
# 뱀게임 AI - 시각화 도구 모음
# 학습 과정 분석 및 결과 시각화 전용 모듈
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import os

# 다크모드 색상 설정
class DarkColors:
    """
    다크모드에 최적화된 시각화 색상 클래스
    
    모든 시각화에 일관된 다크테마 적용
    """
    PRIMARY = '#39D353'        # Spotify Green
    SECONDARY = '#58A6FF'      # Blue
    ACCENT = '#F85149'         # Red
    WARNING = '#FFD33D'        # Yellow
    SUCCESS = '#57D353'        # Light Green
    INFO = '#79C0FF'          # Light Blue
    
    BACKGROUND = '#0D1117'     # Dark Background
    SURFACE = '#161B22'        # Surface Background
    TEXT = '#F0F6FC'          # Primary Text
    MUTED = '#8B949E'         # Muted Text
    
    # 그래프용 다채로운 색상 팔레트
    GRAPH_COLORS = [
        '#39D353', '#F85149', '#58A6FF', '#FFD33D', 
        '#FF7B72', '#79C0FF', '#A5A5A5', '#FFA657',
        '#F778BA', '#7EE787', '#FBD2CC', '#B392F0'
    ]

def setup_dark_theme():
    """
    Matplotlib과 Seaborn을 다크모드로 설정하는 함수
    
    Returns:
        None
        
    설명:
        - 모든 시각화에 일관된 다크테마 적용
        - 텍스트, 배경, 격자 등 모든 요소 다크모드 최적화
        - 폰트 크기 및 스타일 조정
    """
    # Matplotlib 다크 테마 설정
    plt.style.use('dark_background')
    
    # 상세 설정 변경
    plt.rcParams.update({
        'figure.facecolor': DarkColors.BACKGROUND,
        'axes.facecolor': DarkColors.SURFACE,
        'axes.edgecolor': DarkColors.MUTED,
        'axes.labelcolor': DarkColors.TEXT,
        'axes.titlecolor': DarkColors.TEXT,
        'text.color': DarkColors.TEXT,
        'xtick.color': DarkColors.TEXT,
        'ytick.color': DarkColors.TEXT,
        'grid.color': '#30363D',
        'grid.alpha': 0.3,
        'figure.figsize': (12, 8),
        'font.size': 11,
        'axes.titlesize': 15,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'legend.facecolor': DarkColors.SURFACE,
        'legend.edgecolor': DarkColors.MUTED
    })
    
    # Seaborn 색상 팔레트 설정
    sns.set_palette(DarkColors.GRAPH_COLORS)

class TrainingVisualizer:
    """
    Q-Learning 학습 과정을 시각화하는 고급 클래스
    
    주요 기능:
    - 학습 진행 상황 종합 시각화
    - Q값 히트맵 생성
    - 성능 메트릭 분석
    - 학습 곡선 비교
    - 실시간 모니터링 지원
    """
    
    def __init__(self):
        """시각화 도구 초기화"""
        setup_dark_theme()
        self.colors = DarkColors()
        
        print("📊 TrainingVisualizer 초기화 완료 (다크모드)")
    
    def plot_training_progress(self, agent, save_path=None, show_plot=True):
        """
        학습 진행 상황을 종합적으로 시각화하는 함수
        
        Args:
            agent: Q-Learning 에이전트 (학습 기록 포함)
            save_path (str, optional): 그래프 저장 경로
            show_plot (bool): 그래프 표시 여부
            
        Returns:
            None
            
        설명:
            - 점수, 보상, 탐험율, Q-테이블 크기를 4개 서브플롯으로 표시
            - 이동평균으로 트렌드 파악 용이
            - 다크모드에 최적화된 색상과 스타일 사용
        """
        if not agent.training_history['scores']:
            print("⚠️ 학습 기록이 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🧠 Q-Learning 학습 진행 상황', 
                     fontsize=20, fontweight='bold', 
                     color=self.colors.TEXT)
        
        episodes = range(1, len(agent.training_history['scores']) + 1)
        
        # 1. 점수 변화 (좌상단)
        self._plot_scores(axes[0, 0], agent.training_history['scores'], episodes)
        
        # 2. 보상 변화 (우상단)
        self._plot_rewards(axes[0, 1], agent.training_history['rewards'], episodes)
        
        # 3. 탐험율 변화 (좌하단)
        self._plot_epsilon(axes[1, 0], agent.training_history['epsilons'], episodes)
        
        # 4. Q-테이블 크기 변화 (우하단)
        self._plot_qtable_size(axes[1, 1], agent.training_history['q_table_size'], episodes)
        
        plt.tight_layout()
        
        # 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors.BACKGROUND, edgecolor='none')
            print(f"📊 학습 진행 그래프 저장: {save_path}")
        
        # 표시
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def _plot_scores(self, ax, scores, episodes):
        """점수 변화 그래프"""
        ax.plot(episodes, scores, color=self.colors.PRIMARY, 
                alpha=0.6, linewidth=1, label='에피소드별 점수')
        
        # 이동평균 추가
        window = min(50, len(scores) // 10)
        if len(scores) >= window and window > 1:
            moving_avg = self._calculate_moving_average(scores, window)
            ax.plot(episodes[window-1:], moving_avg, color=self.colors.ACCENT, 
                    linewidth=3, label=f'이동평균 ({window} 에피소드)')
        
        # 목표 점수 라인 (있는 경우)
        if hasattr(self, 'target_score'):
            ax.axhline(y=self.target_score, color=self.colors.WARNING, 
                      linestyle='--', alpha=0.8, label=f'목표: {self.target_score}')
        
        ax.set_title('🎯 게임 점수 변화', fontweight='bold', color=self.colors.TEXT)
        ax.set_xlabel('에피소드')
        ax.set_ylabel('점수')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        max_score = max(scores)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        ax.text(0.02, 0.98, f'최고: {max_score}\n평균: {avg_score:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
    
    def _plot_rewards(self, ax, rewards, episodes):
        """보상 변화 그래프"""
        ax.plot(episodes, rewards, color=self.colors.SECONDARY, 
                alpha=0.6, linewidth=1, label='에피소드별 총 보상')
        
        # 이동평균
        window = min(50, len(rewards) // 10)
        if len(rewards) >= window and window > 1:
            moving_avg = self._calculate_moving_average(rewards, window)
            ax.plot(episodes[window-1:], moving_avg, color=self.colors.WARNING, 
                    linewidth=3, label=f'이동평균 ({window} 에피소드)')
        
        ax.set_title('💰 총 보상 변화', fontweight='bold', color=self.colors.TEXT)
        ax.set_xlabel('에피소드')
        ax.set_ylabel('총 보상')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 통계 정보
        max_reward = max(rewards)
        avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        ax.text(0.02, 0.98, f'최고: {max_reward:.1f}\n평균: {avg_reward:.1f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
    
    def _plot_epsilon(self, ax, epsilons, episodes):
        """탐험율 변화 그래프"""
        ax.plot(episodes, epsilons, color=self.colors.ACCENT, 
                linewidth=2, label='Epsilon (탐험율)')
        
        # 구간 표시
        if len(epsilons) > 0:
            # 탐험 단계 (높은 epsilon)
            ax.axhspan(0.5, 1.0, alpha=0.2, color=self.colors.ACCENT, label='탐험 단계')
            # 활용 단계 (낮은 epsilon)  
            ax.axhspan(0.0, 0.1, alpha=0.2, color=self.colors.PRIMARY, label='활용 단계')
        
        ax.set_title('🎲 탐험율 변화', fontweight='bold', color=self.colors.TEXT)
        ax.set_xlabel('에피소드')
        ax.set_ylabel('Epsilon')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 현재 상태 표시
        if epsilons:
            current_epsilon = epsilons[-1]
            stage = "탐험 중" if current_epsilon > 0.1 else "활용 중"
            ax.text(0.98, 0.98, f'현재: {current_epsilon:.3f}\n({stage})', 
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
    
    def _plot_qtable_size(self, ax, q_sizes, episodes):
        """Q-테이블 크기 변화 그래프"""
        ax.plot(episodes, q_sizes, color=self.colors.WARNING, 
                linewidth=2, label='탐험한 상태 수')
        
        # 성장률 계산
        if len(q_sizes) > 10:
            growth_rate = (q_sizes[-1] - q_sizes[0]) / len(q_sizes)
            ax.text(0.02, 0.98, f'성장률: {growth_rate:.1f}/ep', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
        
        ax.set_title('🧠 Q-테이블 크기 변화', fontweight='bold', color=self.colors.TEXT)
        ax.set_xlabel('에피소드')
        ax.set_ylabel('상태 수')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 최종 크기 표시
        if q_sizes:
            final_size = q_sizes[-1]
            ax.text(0.98, 0.02, f'최종: {final_size:,}개 상태', 
                    transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
    
    def _calculate_moving_average(self, data, window):
        """이동평균 계산"""
        if len(data) < window:
            return []
        
        moving_avg = []
        for i in range(window - 1, len(data)):
            avg = np.mean(data[i - window + 1:i + 1])
            moving_avg.append(avg)
        return moving_avg
    
    def plot_q_value_heatmap(self, agent, sample_states=20, save_path=None, show_plot=True):
        """
        Q값을 히트맵으로 시각화하는 함수
        
        Args:
            agent: Q-Learning 에이전트
            sample_states (int): 샘플링할 상태 수
            save_path (str, optional): 저장 경로
            show_plot (bool): 그래프 표시 여부
            
        Returns:
            None
        """
        if len(agent.q_table) == 0:
            print("⚠️ Q-테이블이 비어있습니다.")
            return
        
        # 상태 샘플링
        states = list(agent.q_table.keys())
        if len(states) > sample_states:
            sampled_states = random.sample(states, sample_states)
        else:
            sampled_states = states
        
        # Q값 매트릭스 구성
        q_matrix = []
        state_labels = []
        
        for state_key in sampled_states:
            q_values = agent.q_table[state_key]
            q_matrix.append(q_values)
            # 상태 라벨 축약
            short_label = state_key[:15] + '...' if len(state_key) > 15 else state_key
            state_labels.append(short_label)
        
        q_matrix = np.array(q_matrix)
        
        # 히트맵 생성
        plt.figure(figsize=(10, max(8, len(sampled_states) * 0.4)))
        
        # 색상맵: 음수는 빨강, 양수는 초록
        sns.heatmap(
            q_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',  # 빨강-노랑-초록
            center=0,
            xticklabels=['직진', '우회전', '좌회전'],
            yticklabels=state_labels,
            cbar_kws={'label': 'Q값'},
            linewidths=0.5,
            linecolor=self.colors.MUTED
        )
        
        plt.title('🎯 주요 상태별 Q값 분포', 
                 fontsize=16, fontweight='bold', 
                 color=self.colors.TEXT)
        
        plt.xlabel('행동', fontsize=12, color=self.colors.TEXT)
        plt.ylabel('게임 상태', fontsize=12, color=self.colors.TEXT)
        
        # 통계 정보 추가
        max_q = np.max(q_matrix)
        min_q = np.min(q_matrix)
        avg_q = np.mean(q_matrix)
        
        plt.figtext(0.02, 0.02, f'Q값 범위: {min_q:.2f} ~ {max_q:.2f} (평균: {avg_q:.2f})', 
                   fontsize=10, color=self.colors.MUTED)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.colors.BACKGROUND, edgecolor='none')
            print(f"📊 Q값 히트맵 저장: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_performance_comparison(self, training_histories, labels, save_path=None, show_plot=True):
        """
        여러 학습 결과를 비교하는 함수
        
        Args:
            training_histories (list): 학습 기록들의 리스트
            labels (list): 각 학습의 라벨
            save_path (str, optional): 저장 경로
            show_plot (bool): 그래프 표시 여부
            
        Returns:
            None
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📊 학습 성능 비교', 
                     fontsize=20, fontweight='bold', 
                     color=self.colors.TEXT)
        
        colors = self.colors.GRAPH_COLORS[:len(training_histories)]
        
        for i, (history, label, color) in enumerate(zip(training_histories, labels, colors)):
            episodes = range(1, len(history['scores']) + 1)
            
            # 점수 비교
            axes[0, 0].plot(episodes, history['scores'], 
                           color=color, alpha=0.7, label=label)
            
            # 보상 비교
            axes[0, 1].plot(episodes, history['rewards'], 
                           color=color, alpha=0.7, label=label)
            
            # 탐험율 비교
            axes[1, 0].plot(episodes, history['epsilons'], 
                           color=color, alpha=0.7, label=label)
            
            # Q-테이블 크기 비교
            axes[1, 1].plot(episodes, history['q_table_size'], 
                           color=color, alpha=0.7, label=label)
        
        # 각 서브플롯 설정
        titles = ['🎯 점수 비교', '💰 보상 비교', '🎲 탐험율 비교', '🧠 Q-테이블 크기 비교']
        ylabels = ['점수', '총 보상', 'Epsilon', '상태 수']
        
        for ax, title, ylabel in zip(axes.flat, titles, ylabels):
            ax.set_title(title, fontweight='bold', color=self.colors.TEXT)
            ax.set_xlabel('에피소드')
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.colors.BACKGROUND, edgecolor='none')
            print(f"📊 성능 비교 그래프 저장: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_curve_analysis(self, agent, save_path=None, show_plot=True):
        """
        학습 곡선 상세 분석
        
        Args:
            agent: Q-Learning 에이전트
            save_path (str, optional): 저장 경로  
            show_plot (bool): 그래프 표시 여부
            
        Returns:
            dict: 분석 결과
        """
        scores = agent.training_history['scores']
        if not scores:
            print("⚠️ 학습 기록이 없습니다.")
            return None
        
        # 분석 지표 계산
        episodes = np.array(range(1, len(scores) + 1))
        
        # 구간별 성능 분석
        n_segments = 5
        segment_size = len(scores) // n_segments
        segment_avgs = []
        
        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size if i < n_segments - 1 else len(scores)
            segment_avg = np.mean(scores[start:end])
            segment_avgs.append(segment_avg)
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📈 학습 곡선 상세 분석', 
                     fontsize=20, fontweight='bold', 
                     color=self.colors.TEXT)
        
        # 1. 구간별 성능 개선
        segment_labels = [f'구간 {i+1}' for i in range(n_segments)]
        bars = axes[0, 0].bar(segment_labels, segment_avgs, 
                             color=self.colors.GRAPH_COLORS[:n_segments])
        axes[0, 0].set_title('📊 구간별 평균 성능', fontweight='bold')
        axes[0, 0].set_ylabel('평균 점수')
        
        # 값 표시
        for bar, avg in zip(bars, segment_avgs):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{avg:.1f}', ha='center', va='bottom')
        
        # 2. 성능 향상률
        if len(segment_avgs) > 1:
            improvement_rates = []
            for i in range(1, len(segment_avgs)):
                rate = (segment_avgs[i] - segment_avgs[i-1]) / segment_avgs[i-1] * 100
                improvement_rates.append(rate)
            
            x_pos = range(1, len(segment_avgs))
            colors = [self.colors.PRIMARY if rate >= 0 else self.colors.ACCENT 
                     for rate in improvement_rates]
            
            bars = axes[0, 1].bar(x_pos, improvement_rates, color=colors)
            axes[0, 1].set_title('📈 구간별 성능 향상률', fontweight='bold')
            axes[0, 1].set_ylabel('향상률 (%)')
            axes[0, 1].set_xlabel('구간 전환')
            axes[0, 1].axhline(y=0, color=self.colors.MUTED, linestyle='-', alpha=0.5)
            
            # 값 표시
            for bar, rate in zip(bars, improvement_rates):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + (1 if rate >= 0 else -3),
                               f'{rate:+.1f}%', ha='center', 
                               va='bottom' if rate >= 0 else 'top')
        
        # 3. 점수 분포 히스토그램
        axes[1, 0].hist(scores, bins=20, alpha=0.7, color=self.colors.SECONDARY, 
                       edgecolor=self.colors.MUTED)
        axes[1, 0].axvline(np.mean(scores), color=self.colors.ACCENT, 
                          linestyle='--', linewidth=2, label=f'평균: {np.mean(scores):.1f}')
        axes[1, 0].axvline(np.median(scores), color=self.colors.WARNING, 
                          linestyle='--', linewidth=2, label=f'중간값: {np.median(scores):.1f}')
        axes[1, 0].set_title('📊 점수 분포', fontweight='bold')
        axes[1, 0].set_xlabel('점수')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].legend()
        
        # 4. 수렴성 분석 (최근 100 에피소드 변동성)
        if len(scores) >= 100:
            recent_scores = scores[-100:]
            rolling_std = []
            window = 10
            
            for i in range(window, len(recent_scores) + 1):
                std = np.std(recent_scores[i-window:i])
                rolling_std.append(std)
            
            x_pos = range(len(rolling_std))
            axes[1, 1].plot(x_pos, rolling_std, color=self.colors.INFO, linewidth=2)
            axes[1, 1].set_title('📉 학습 안정성 (최근 100 에피소드)', fontweight='bold')
            axes[1, 1].set_xlabel('에피소드 (최근 100개 중)')
            axes[1, 1].set_ylabel('점수 표준편차')
            
            # 수렴 여부 판단
            final_std = np.mean(rolling_std[-10:]) if len(rolling_std) >= 10 else rolling_std[-1]
            convergence_status = "수렴됨" if final_std < 2.0 else "수렴 중"
            axes[1, 1].text(0.98, 0.98, f'상태: {convergence_status}\n변동성: {final_std:.2f}', 
                            transform=axes[1, 1].transAxes, 
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor=self.colors.SURFACE, alpha=0.8))
        
        plt.tight_layout()
        
        # 분석 결과 반환
        analysis_result = {
            'segment_averages': segment_avgs,
            'improvement_rates': improvement_rates if len(segment_avgs) > 1 else [],
            'overall_average': np.mean(scores),
            'best_score': np.max(scores),
            'score_std': np.std(scores),
            'convergence_status': convergence_status if len(scores) >= 100 else "분석 불가"
        }
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor=self.colors.BACKGROUND, edgecolor='none')
            print(f"📊 학습 곡선 분석 저장: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return analysis_result

# 편의 함수들
def plot_training_results(agent, save_dir="logs", show_plot=True):
    """
    학습 결과를 종합적으로 시각화하는 편의 함수
    
    Args:
        agent: Q-Learning 에이전트
        save_dir (str): 저장 디렉토리
        show_plot (bool): 그래프 표시 여부
        
    Returns:
        None
    """
    visualizer = TrainingVisualizer()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 기본 학습 진행 그래프
    progress_path = os.path.join(save_dir, f"training_progress_{timestamp}.png")
    visualizer.plot_training_progress(agent, save_path=progress_path, show_plot=show_plot)
    
    # Q값 히트맵
    heatmap_path = os.path.join(save_dir, f"q_values_heatmap_{timestamp}.png")
    visualizer.plot_q_value_heatmap(agent, save_path=heatmap_path, show_plot=show_plot)
    
    # 학습 곡선 분석
    analysis_path = os.path.join(save_dir, f"learning_analysis_{timestamp}.png")
    analysis_result = visualizer.plot_learning_curve_analysis(agent, save_path=analysis_path, show_plot=show_plot)
    
    print(f"📊 모든 시각화 완료! 저장 위치: {save_dir}")
    return analysis_result

def create_heatmap(data, labels=None, title="히트맵", save_path=None):
    """
    범용 히트맵 생성 함수
    
    Args:
        data: 2D 배열 데이터
        labels: 축 라벨들
        title: 그래프 제목
        save_path: 저장 경로
        
    Returns:
        None
    """
    setup_dark_theme()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=labels[0] if labels else True,
                yticklabels=labels[1] if labels else True)
    
    plt.title(title, fontsize=16, fontweight='bold', color=DarkColors.TEXT)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor=DarkColors.BACKGROUND)
        print(f"📊 히트맵 저장: {save_path}")
    
    plt.show()

# 초기화
setup_dark_theme()
print("📊 utils.visualization 모듈이 로드되었습니다!")