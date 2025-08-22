"""
프레임 추출 모듈 테스트
Tests for frame extraction module
"""
import unittest
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np

# 테스트를 위한 경로 설정
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.video_processor.frame_extractor import FrameExtractor

class TestFrameExtractor(unittest.TestCase):
    """FrameExtractor 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = FrameExtractor(output_dir=self.temp_dir)
        self.test_video_path = None
        
    def tearDown(self):
        """테스트 정리"""
        # 테스트 파일들 정리
        if self.test_video_path and os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
    
    def create_test_video(self, duration_seconds=5, fps=30):
        """테스트용 비디오 파일 생성"""
        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_video.close()
        
        # OpenCV를 사용해 테스트 비디오 생성
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, fps, (640, 480))
        
        total_frames = duration_seconds * fps
        for i in range(total_frames):
            # 간단한 패턴의 프레임 생성
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # 프레임마다 다른 색상으로 원 그리기
            color = (i % 255, (i * 2) % 255, (i * 3) % 255)
            center = (320 + int(50 * np.sin(i * 0.1)), 240 + int(50 * np.cos(i * 0.1)))
            cv2.circle(frame, center, 50, color, -1)
            
            # 프레임 번호 텍스트 추가
            cv2.putText(frame, f'Frame {i}', (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        self.test_video_path = temp_video.name
        return temp_video.name
    
    def test_extract_frames_basic(self):
        """기본 프레임 추출 테스트"""
        video_path = self.create_test_video(duration_seconds=3, fps=30)
        
        # 1초 간격으로 프레임 추출
        extracted_frames = self.extractor.extract_frames(
            video_path, 
            interval_seconds=1
        )
        
        # 3초 비디오에서 1초 간격이므로 약 3-4개 프레임 추출 예상
        self.assertGreaterEqual(len(extracted_frames), 2)
        self.assertLessEqual(len(extracted_frames), 4)
        
        # 추출된 파일들이 실제로 존재하는지 확인
        for frame_path in extracted_frames:
            self.assertTrue(os.path.exists(frame_path))
            self.assertTrue(frame_path.endswith('.jpg'))
    
    def test_extract_frames_with_max_limit(self):
        """최대 프레임 수 제한 테스트"""
        video_path = self.create_test_video(duration_seconds=5, fps=30)
        
        # 최대 2개만 추출
        extracted_frames = self.extractor.extract_frames(
            video_path,
            interval_seconds=1,
            max_frames=2
        )
        
        self.assertEqual(len(extracted_frames), 2)
    
    def test_get_video_info(self):
        """비디오 정보 추출 테스트"""
        video_path = self.create_test_video(duration_seconds=3, fps=30)
        
        video_info = self.extractor.get_video_info(video_path)
        
        self.assertIn('fps', video_info)
        self.assertIn('frame_count', video_info)
        self.assertIn('width', video_info)
        self.assertIn('height', video_info)
        self.assertIn('duration', video_info)
        
        # 예상 값 확인
        self.assertAlmostEqual(video_info['fps'], 30, delta=1)
        self.assertEqual(video_info['width'], 640)
        self.assertEqual(video_info['height'], 480)
        self.assertAlmostEqual(video_info['duration'], 3, delta=0.5)
    
    def test_invalid_video_path(self):
        """존재하지 않는 비디오 파일 테스트"""
        with self.assertRaises(FileNotFoundError):
            self.extractor.extract_frames('/nonexistent/video.mp4')
    
    def test_extract_frames_by_timestamp(self):
        """특정 타임스탬프에서 프레임 추출 테스트"""
        video_path = self.create_test_video(duration_seconds=5, fps=30)
        
        timestamps = [1.0, 2.5, 4.0]  # 1초, 2.5초, 4초 지점
        extracted_frames = self.extractor.extract_frames_by_timestamp(
            video_path, 
            timestamps
        )
        
        self.assertEqual(len(extracted_frames), 3)
        
        # 파일 존재 확인
        for frame_path in extracted_frames:
            self.assertTrue(os.path.exists(frame_path))

class TestFrameValidation(unittest.TestCase):
    """프레임 유효성 검사 테스트"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.extractor = FrameExtractor(output_dir=self.temp_dir)
    
    def test_valid_frame(self):
        """유효한 프레임 테스트"""
        # 밝고 선명한 테스트 프레임 생성
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)
        
        self.assertTrue(self.extractor._is_frame_valid(frame))
    
    def test_dark_frame(self):
        """너무 어두운 프레임 테스트"""
        # 매우 어두운 프레임
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 10
        
        self.assertFalse(self.extractor._is_frame_valid(frame))
    
    def test_blurry_frame(self):
        """흐릿한 프레임 테스트"""
        # 흐릿한 프레임 (가우시안 블러 적용)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame = cv2.GaussianBlur(frame, (51, 51), 20)
        
        self.assertFalse(self.extractor._is_frame_valid(frame))
    
    def test_empty_frame(self):
        """빈 프레임 테스트"""
        self.assertFalse(self.extractor._is_frame_valid(None))
        self.assertFalse(self.extractor._is_frame_valid(np.array([])))

if __name__ == '__main__':
    unittest.main()