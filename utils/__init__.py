# 📦 utils/__init__.py
# ================================================================
# 유틸리티 패키지 초기화 파일
# 헬퍼 함수들과 시각화 도구들
# ================================================================

"""
🛠️ Utils Package

프로젝트 전반에서 사용되는 유틸리티 함수들과 도구들을 제공합니다.

주요 구성요소:
- 시각화 도구
- 설정 관리
- 데이터 처리 헬퍼
- 로깅 도구
"""

__version__ = "1.0.0"

# 시각화 도구들 import
from .visualization import (
    TrainingVisualizer,
    plot_training_results,
    create_heatmap,
    setup_dark_theme
)

# 유틸리티 함수들
def create_directory(path):
    """디렉토리 생성 (존재하지 않는 경우)"""
    import os
    os.makedirs(path, exist_ok=True)
    print(f"📁 디렉토리 생성: {path}")

def get_timestamp():
    """현재 시간 문자열 반환"""
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_json(data, filepath):
    """JSON 파일 저장"""
    import json
    import os
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"💾 JSON 저장: {filepath}")

def load_json(filepath):
    """JSON 파일 로드"""
    import json
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"📂 JSON 로드: {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없음: {filepath}")
        return None

def setup_logging(log_dir="logs"):
    """로깅 설정"""
    import logging
    import os
    
    # 로그 디렉토리 생성
    os.makedirs(log_dir, exist_ok=True)
    
    # 로그 파일명
    log_file = os.path.join(log_dir, f"training_{get_timestamp()}.log")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    print(f"📝 로깅 설정 완료: {log_file}")
    return logging.getLogger(__name__)

# 패키지에서 export할 함수들
__all__ = [
    'create_directory',
    'get_timestamp', 
    'save_json',
    'load_json',
    'setup_logging',
    'TrainingVisualizer',
    'plot_training_results',
    'create_heatmap',
    'setup_dark_theme'
]

# 패키지 초기화
print("🛠️ Utils package loaded!")