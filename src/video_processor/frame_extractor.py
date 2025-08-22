"""
CCTV 영상에서 프레임을 추출하는 모듈
Frame extraction module for CCTV footage analysis
"""
import cv2
import os
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta

class FrameExtractor:
    """CCTV 영상에서 일정 간격으로 프레임을 추출하는 클래스"""
    
    def __init__(self, output_dir: str = "output/extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def extract_frames(self, video_path: str, interval_seconds: int = 1, 
                      max_frames: Optional[int] = None) -> List[str]:
        """
        비디오에서 지정된 간격으로 프레임을 추출
        
        Args:
            video_path: 비디오 파일 경로
            interval_seconds: 프레임 추출 간격 (초)
            max_frames: 최대 추출할 프레임 수 (None이면 전체)
            
        Returns:
            추출된 프레임 파일 경로 리스트
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        # 비디오 정보 추출
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        self.logger.info(f"비디오 정보: FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.2f}초")
        
        # 프레임 간격 계산
        frame_interval = int(fps * interval_seconds)
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        # 비디오 파일명에서 확장자 제거
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 지정된 간격마다 프레임 저장
            if frame_count % frame_interval == 0:
                # 프레임 품질 확인
                if self._is_frame_valid(frame):
                    # 현재 시간 계산
                    current_time = frame_count / fps
                    time_str = self._seconds_to_timestamp(current_time)
                    
                    # 파일명 생성
                    filename = f"{video_name}_{timestamp}_frame_{saved_count:06d}_{time_str}.jpg"
                    filepath = self.output_dir / filename
                    
                    # 프레임 저장
                    cv2.imwrite(str(filepath), frame)
                    extracted_frames.append(str(filepath))
                    saved_count += 1
                    
                    self.logger.info(f"프레임 저장: {filename} (시간: {time_str})")
                    
                    # 최대 프레임 수 체크
                    if max_frames and saved_count >= max_frames:
                        break
            
            frame_count += 1
        
        cap.release()
        self.logger.info(f"총 {saved_count}개의 프레임을 추출했습니다")
        
        return extracted_frames
    
    def extract_frames_by_timestamp(self, video_path: str, timestamps: List[float]) -> List[str]:
        """
        특정 타임스탬프에서 프레임을 추출
        
        Args:
            video_path: 비디오 파일 경로
            timestamps: 추출할 시간 리스트 (초 단위)
            
        Returns:
            추출된 프레임 파일 경로 리스트
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        extracted_frames = []
        
        for i, time_sec in enumerate(timestamps):
            # 해당 시간의 프레임으로 이동
            frame_number = int(time_sec * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if ret and self._is_frame_valid(frame):
                time_str = self._seconds_to_timestamp(time_sec)
                filename = f"{video_name}_{timestamp}_timestamp_{i:03d}_{time_str}.jpg"
                filepath = self.output_dir / filename
                
                cv2.imwrite(str(filepath), frame)
                extracted_frames.append(str(filepath))
                
                self.logger.info(f"타임스탬프 프레임 저장: {filename}")
        
        cap.release()
        return extracted_frames
    
    def extract_motion_frames(self, video_path: str, motion_threshold: float = 0.1,
                            max_frames: Optional[int] = None) -> List[str]:
        """
        움직임이 감지된 프레임만 추출
        
        Args:
            video_path: 비디오 파일 경로
            motion_threshold: 움직임 감지 임계값
            max_frames: 최대 추출할 프레임 수
            
        Returns:
            추출된 프레임 파일 경로 리스트
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = Path(video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = frame_count / fps
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # 프레임 차이 계산
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = np.mean(diff) / 255.0
                
                # 움직임이 임계값을 초과하면 저장
                if mean_diff > motion_threshold:
                    if self._is_frame_valid(frame):
                        time_str = self._seconds_to_timestamp(current_time)
                        filename = f"{video_name}_{timestamp}_motion_{saved_count:06d}_{time_str}.jpg"
                        filepath = self.output_dir / filename
                        
                        cv2.imwrite(str(filepath), frame)
                        extracted_frames.append(str(filepath))
                        saved_count += 1
                        
                        self.logger.info(f"움직임 프레임 저장: {filename} (변화량: {mean_diff:.3f})")
                        
                        if max_frames and saved_count >= max_frames:
                            break
            
            prev_frame = gray.copy()
            frame_count += 1
        
        cap.release()
        self.logger.info(f"총 {saved_count}개의 움직임 프레임을 추출했습니다")
        
        return extracted_frames
    
    def get_video_info(self, video_path: str) -> dict:
        """비디오 파일의 메타데이터 정보를 반환"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'duration_formatted': self._seconds_to_timestamp(duration)
        }
    
    def _is_frame_valid(self, frame: np.ndarray) -> bool:
        """프레임이 유효한지 확인 (너무 어둡거나 흐릿하지 않은지)"""
        if frame is None or frame.size == 0:
            return False
        
        # 밝기 체크
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # 너무 어두우면 제외
        if brightness < 20:
            return False
        
        # 블러 체크 (라플라시안 분산)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 너무 흐리면 제외
        if laplacian_var < 100:
            return False
        
        return True
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """초를 HH:MM:SS 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}_{minutes:02d}_{secs:02d}"
    
    def batch_extract(self, video_directory: str, interval_seconds: int = 1,
                     max_frames_per_video: Optional[int] = None) -> dict:
        """
        디렉터리 내 모든 비디오 파일에서 프레임을 일괄 추출
        
        Args:
            video_directory: 비디오 파일들이 있는 디렉터리
            interval_seconds: 프레임 추출 간격
            max_frames_per_video: 비디오당 최대 추출할 프레임 수
            
        Returns:
            비디오별 추출된 프레임 정보
        """
        video_dir = Path(video_directory)
        if not video_dir.exists():
            raise FileNotFoundError(f"디렉터리를 찾을 수 없습니다: {video_directory}")
        
        # 지원하는 비디오 형식
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
        
        results = {}
        
        for video_file in video_files:
            self.logger.info(f"비디오 처리 시작: {video_file.name}")
            try:
                extracted_frames = self.extract_frames(
                    str(video_file), 
                    interval_seconds, 
                    max_frames_per_video
                )
                video_info = self.get_video_info(str(video_file))
                
                results[video_file.name] = {
                    'extracted_frames': extracted_frames,
                    'frame_count': len(extracted_frames),
                    'video_info': video_info,
                    'status': 'success'
                }
                
            except Exception as e:
                self.logger.error(f"비디오 처리 실패 {video_file.name}: {str(e)}")
                results[video_file.name] = {
                    'extracted_frames': [],
                    'frame_count': 0,
                    'video_info': {},
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results