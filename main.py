#!/usr/bin/env python3
"""
SafetyVision AI - 탱크로리 안전상태 체크 시스템
Main application entry point for tanker truck safety monitoring system
"""
import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.web_interface.app import SafetyVisionApp

def setup_logging():
    """로깅 설정"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 로그 디렉토리 생성
    log_dir = project_root / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 파일과 콘솔 모두에 로깅
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_dir / 'safetyvision.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """필수 의존성 패키지 확인"""
    required_packages = [
        ('flask', 'flask'), 
        ('cv2', 'opencv-python'), 
        ('ultralytics', 'ultralytics'), 
        ('torch', 'torch'), 
        ('torchvision', 'torchvision'),
        ('numpy', 'numpy'), 
        ('PIL', 'pillow'), 
        ('matplotlib', 'matplotlib'), 
        ('seaborn', 'seaborn'), 
        ('pandas', 'pandas'),
        ('dotenv', 'python-dotenv'), 
        ('werkzeug', 'werkzeug'), 
        ('jinja2', 'jinja2'), 
        ('reportlab', 'reportlab')
    ]
    
    missing_packages = []
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        logging.error(f"다음 패키지들을 설치해주세요: {', '.join(missing_packages)}")
        logging.error("pip install -r requirements.txt 명령을 실행하세요.")
        # 개발 환경에서는 경고만 출력하고 계속 진행
        if os.getenv('FLASK_ENV') == 'development':
            logging.warning("개발 환경이므로 계속 진행합니다...")
        else:
            sys.exit(1)

def check_directories():
    """필요한 디렉토리들이 존재하는지 확인하고 생성"""
    required_dirs = [
        'uploads/videos',
        'uploads/images', 
        'output/reports',
        'output/annotated_images',
        'output/extracted_frames',
        'models',
        'logs'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"디렉토리 확인: {full_path}")

def print_banner():
    """애플리케이션 시작 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║                    SafetyVision AI                           ║
    ║                                                              ║
    ║              탱크로리 안전상태 체크 시스템                   ║
    ║                                                              ║
    ║    AI 기반 CCTV 영상 분석을 통한 작업자 안전장비 검출        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """메인 함수"""
    try:
        # 배너 출력
        print_banner()
        
        # 로깅 설정
        setup_logging()
        logging.info("SafetyVision AI 시스템을 시작합니다...")
        
        # 의존성 확인
        logging.info("필수 패키지 의존성을 확인하는 중...")
        check_dependencies()
        logging.info("모든 필수 패키지가 설치되어 있습니다.")
        
        # 디렉토리 확인
        logging.info("필요한 디렉토리들을 확인하는 중...")
        check_directories()
        logging.info("모든 필요한 디렉토리가 준비되었습니다.")
        
        # 환경 변수 확인
        if not os.getenv('SECRET_KEY'):
            logging.warning("SECRET_KEY 환경 변수가 설정되지 않았습니다. 기본값을 사용합니다.")
        
        # Flask 애플리케이션 생성 및 실행
        logging.info("Flask 웹 애플리케이션을 초기화하는 중...")
        app = SafetyVisionApp()
        
        # 개발/프로덕션 모드 결정
        debug_mode = os.getenv('FLASK_ENV') == 'development'
        host = os.getenv('FLASK_HOST', '0.0.0.0')
        port = int(os.getenv('FLASK_PORT', 5000))
        
        logging.info(f"웹 서버를 시작합니다: http://{host}:{port}")
        logging.info(f"디버그 모드: {'활성화' if debug_mode else '비활성화'}")
        
        if debug_mode:
            logging.info("개발 모드로 실행 중입니다. 프로덕션 환경에서는 FLASK_ENV를 변경하세요.")
        
        # 웹 애플리케이션 실행
        app.run(host=host, port=port, debug=debug_mode)
        
    except KeyboardInterrupt:
        logging.info("사용자에 의해 애플리케이션이 중단되었습니다.")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")
        logging.error("자세한 오류 정보:", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()