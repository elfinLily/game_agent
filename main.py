# 🚀 뱀게임 AI - 프로젝트 메인 진입점 (main.py)
# ================================================================
# 작성자: AI/빅데이터 석사과정생
# 환경: VSCode/Colab (다크모드 최적화)
# 목표: 프로젝트의 모든 기능을 통합하는 메인 메뉴
# ================================================================

import sys
import os
from datetime import datetime

# 패키지 import
try:
    from games.snake.game import SnakeGameAI
    from games.snake.agent import QLearningAgent
    from games.snake import check_dependencies
    from utils import create_directory, get_timestamp, setup_logging, TrainingVisualizer
except ImportError as e:
    print(f"❌ 패키지 import 실패: {e}")
    print("💡 __init__.py 파일들이 올바르게 작성되었는지 확인하세요.")
    sys.exit(1)

def show_banner():
    """프로젝트 배너 출력"""
    banner = """
🐍🤖 =============================================== 🤖🐍
    
         뱀게임 AI 프로젝트 (Q-Learning)
         
    🎯 목표: 강화학습으로 뱀게임 마스터하기
    🧠 알고리즘: Q-Learning → DQN → PPO
    🎨 테마: 다크모드 최적화
    
🐍🤖 =============================================== 🤖🐍
    """
    print(banner)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def show_main_menu():
    """메인 메뉴 출력"""
    menu = """
📋 메인 메뉴
=====================================
1. 🎓 AI 학습 시작
2. 🧪 학습된 AI 테스트  
3. 🎮 게임 환경 데모
4. 📊 학습 결과 분석
5. ⚙️  환경 설정 확인
6. 🔧 패키지 테스트
7. ❌ 종료
=====================================
"""
    print(menu)

def run_training():
    """AI 학습 실행"""
    print("🎓 AI 학습 모드를 시작합니다...")
    
    # 학습 모드 선택
    print("\n학습 모드 선택:")
    print("1. 🚀 빠른 데모 (500 에피소드)")
    print("2. 📚 기본 학습 (3000 에피소드)")  
    print("3. 🔥 고급 학습 (10000 에피소드)")
    print("4. 🎛️  커스텀 설정")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == '1':
        from games.snake.train import quick_demo
        return quick_demo()
    elif choice == '2':
        from games.snake.train import main_training
        return main_training()
    elif choice == '3':
        from games.snake.train import advanced_training
        return advanced_training()
    elif choice == '4':
        return custom_training()
    else:
        print("❌ 잘못된 선택입니다.")
        return None

def run_testing():
    """학습된 AI 테스트"""
    print("🧪 AI 테스트 모드를 시작합니다...")
    
    # 모델 파일 찾기
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"❌ {models_dir} 디렉토리가 없습니다.")
        print("💡 먼저 AI 학습을 진행해주세요.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print("❌ 학습된 모델이 없습니다.")
        print("💡 먼저 AI 학습을 진행해주세요.")
        return
    
    print(f"\n📂 사용 가능한 모델 ({len(model_files)}개):")
    for i, model_file in enumerate(model_files, 1):
        print(f"{i}. {model_file}")
    
    try:
        choice = int(input(f"\n모델 선택 (1-{len(model_files)}): ")) - 1
        if 0 <= choice < len(model_files):
            model_path = os.path.join(models_dir, model_files[choice])
            test_model(model_path)
        else:
            print("❌ 잘못된 선택입니다.")
    except ValueError:
        print("❌ 숫자를 입력해주세요.")

def test_model(model_path):
    """특정 모델 테스트"""
    print(f"🧪 모델 테스트: {model_path}")
    
    try:
        from games.snake.train import SnakeAITrainer
        
        trainer = SnakeAITrainer()
        test_episodes = int(input("테스트 에피소드 수 (기본: 10): ") or "10")
        
        results = trainer.test_trained_agent(model_path, test_episodes)
        
        if results:
            print("\n🎉 테스트 완료!")
            print(f"📊 결과 요약:")
            print(f"   평균 점수: {results['avg_score']:.2f}")
            print(f"   최고 점수: {results['max_score']}")
            print(f"   최저 점수: {results['min_score']}")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

def run_demo():
    """게임 환경 데모"""
    print("🎮 게임 환경 데모를 시작합니다...")
    
    try:
        from games.snake.game import demo_game, test_state_representation
        
        print("\n데모 모드 선택:")
        print("1. 🎮 실제 게임 플레이")
        print("2. 🧪 상태 표현 테스트")
        
        choice = input("선택 (1-2): ").strip()
        
        if choice == '1':
            demo_game()
        elif choice == '2':
            test_state_representation()
        else:
            print("❌ 잘못된 선택입니다.")
            
    except Exception as e:
        print(f"❌ 데모 실행 실패: {e}")

def analyze_results():
    """학습 결과 분석"""
    print("📊 학습 결과 분석을 시작합니다...")
    
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"❌ {logs_dir} 디렉토리가 없습니다.")
        return
    
    log_dirs = [d for d in os.listdir(logs_dir) 
                if os.path.isdir(os.path.join(logs_dir, d))]
    
    if not log_dirs:
        print("❌ 분석할 학습 로그가 없습니다.")
        return
    
    print(f"\n📂 사용 가능한 로그 ({len(log_dirs)}개):")
    for i, log_dir in enumerate(log_dirs, 1):
        print(f"{i}. {log_dir}")
    
    # 간단한 분석 기능 구현
    print("💡 고급 분석 기능은 향후 구현 예정입니다.")

def check_environment():
    """환경 설정 확인"""
    print("⚙️ 환경 설정을 확인합니다...")
    
    # 파이썬 버전
    print(f"🐍 Python 버전: {sys.version}")
    
    # 의존성 확인
    print("\n📦 라이브러리 확인:")
    check_dependencies()
    
    # 디렉토리 확인
    print("\n📁 프로젝트 구조:")
    required_dirs = ['games', 'utils', 'models', 'logs']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (없음)")
            create_directory(dir_name)

def test_packages():
    """패키지 테스트"""
    print("🔧 패키지 기능을 테스트합니다...")
    
    try:
        # snake 패키지 테스트
        print("\n🐍 snake 패키지 테스트:")
        game = SnakeGameAI(display=False)
        agent = QLearningAgent()
        print("✅ 게임 환경 및 에이전트 생성 성공")
        
        # 간단한 동작 테스트
        state = game.get_state()
        action = agent.get_action(state)
        reward, done, score = game.play_step(action)
        print(f"✅ 기본 동작 테스트 성공 (보상: {reward:.2f})")
        
        # utils 패키지 테스트
        print("\n🛠️ utils 패키지 테스트:")
        timestamp = get_timestamp()
        print(f"✅ 타임스탬프 생성: {timestamp}")
        
        print("\n🎉 모든 패키지가 정상 작동합니다!")
        
    except Exception as e:
        print(f"❌ 패키지 테스트 실패: {e}")

def custom_training():
    """커스텀 학습 설정"""
    print("🎛️ 커스텀 학습 설정")
    print("="*40)
    
    try:
        episodes = int(input("에피소드 수 (기본: 1000): ") or "1000")
        target_score = int(input("목표 점수 (기본: 15): ") or "15")
        learning_rate = float(input("학습률 (기본: 0.1): ") or "0.1")
        
        config = {
            'episodes': episodes,
            'target_score': target_score,
            'learning_rate': learning_rate,
            'display_game': True,
            'display_interval': max(50, episodes // 20),
            'save_interval': max(100, episodes // 10)
        }
        
        print(f"\n🎯 커스텀 설정:")
        for key, value in config.items():
            print(f"   {key}: {value}")
        
        confirm = input("\n시작하시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            from games.snake.train import SnakeAITrainer
            trainer = SnakeAITrainer(config)
            return trainer.train()
        
    except ValueError:
        print("❌ 잘못된 입력입니다.")

def main():
    """메인 함수"""
    # 초기 설정
    show_banner()
    
    # 기본 디렉토리 생성
    for directory in ['models', 'logs']:
        create_directory(directory)
    
    # 메인 루프
    while True:
        try:
            show_main_menu()
            choice = input("선택해주세요 (1-7): ").strip()
            
            if choice == '1':
                run_training()
            elif choice == '2':
                run_testing()
            elif choice == '3':
                run_demo()
            elif choice == '4':
                analyze_results()
            elif choice == '5':
                check_environment()
            elif choice == '6':
                test_packages()
            elif choice == '7':
                print("\n👋 프로젝트를 종료합니다. 수고하셨습니다!")
                break
            else:
                print("❌ 잘못된 선택입니다. 1-7 사이의 숫자를 입력해주세요.")
            
            # 계속 진행 여부 확인
            if choice in ['1', '2', '3', '4']:
                input("\n⏸️ 계속하려면 Enter를 누르세요...")
                
        except KeyboardInterrupt:
            print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
            print("👋 프로젝트를 종료합니다.")
            break
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            print("💡 문제가 지속되면 패키지 설정을 확인해주세요.")

if __name__ == "__main__":
    main()