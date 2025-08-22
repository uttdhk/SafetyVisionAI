"""
Configuration file for Safety Vision AI
탱크로리 차량 안전상태 체크 시스템 설정
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Upload directories
UPLOAD_FOLDER = BASE_DIR / 'uploads'
VIDEO_UPLOAD_FOLDER = UPLOAD_FOLDER / 'videos'
IMAGE_UPLOAD_FOLDER = UPLOAD_FOLDER / 'images'

# Output directories
OUTPUT_FOLDER = BASE_DIR / 'output'
REPORTS_FOLDER = OUTPUT_FOLDER / 'reports'
ANNOTATED_IMAGES_FOLDER = OUTPUT_FOLDER / 'annotated_images'

# Models directory
MODELS_FOLDER = BASE_DIR / 'models'

# Create directories if they don't exist
for folder in [UPLOAD_FOLDER, VIDEO_UPLOAD_FOLDER, IMAGE_UPLOAD_FOLDER, 
               OUTPUT_FOLDER, REPORTS_FOLDER, ANNOTATED_IMAGES_FOLDER, MODELS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# Flask configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'safety-vision-ai-secret-key'
    UPLOAD_FOLDER = str(UPLOAD_FOLDER)
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
    
    # Video processing settings
    FRAME_EXTRACTION_INTERVAL = 30  # Extract frame every 30 frames (1 second at 30fps)
    SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # AI Model settings
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # Safety equipment classes
    SAFETY_CLASSES = {
        'helmet': '안전모',
        'no_helmet': '안전모 미착용',
        'safety_vest': '안전조끼',
        'no_safety_vest': '안전조끼 미착용',
        'seatbelt': '안전벨트',
        'no_seatbelt': '안전벨트 미착용',
        'person': '작업자',
        'tanker_truck': '탱크로리'
    }
    
    # Report settings
    REPORT_LANGUAGE = 'ko'
    REPORT_FORMAT = 'pdf'