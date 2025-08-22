"""
안전 검출 모듈 테스트
Tests for safety detection module
"""
import unittest
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
from unittest.mock import Mock, patch

# 테스트를 위한 경로 설정
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ai_analyzer.safety_detector import SafetyDetector

class TestSafetyDetector(unittest.TestCase):
    """SafetyDetector 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        
        # YOLO 모델 로딩을 모킹하여 테스트 속도 향상
        with patch('src.ai_analyzer.safety_detector.YOLO'):
            self.detector = SafetyDetector()
            
        self.test_image_path = None
    
    def tearDown(self):
        """테스트 정리"""
        if self.test_image_path and os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
    
    def create_test_image(self, width=640, height=480):
        """테스트용 이미지 생성"""
        temp_image = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        temp_image.close()
        
        # 테스트 이미지 생성 (작업자와 안전장비를 시뮬레이션)
        image = np.ones((height, width, 3), dtype=np.uint8) * 128
        
        # 사람 모양 (직사각형으로 단순화)
        cv2.rectangle(image, (250, 200), (390, 450), (100, 150, 200), -1)
        
        # 안전모 (노란색 원)
        cv2.circle(image, (320, 180), 30, (0, 255, 255), -1)
        
        # 안전조끼 (형광 녹색 직사각형)
        cv2.rectangle(image, (280, 250), (360, 350), (0, 255, 0), -1)
        
        cv2.imwrite(temp_image.name, image)
        self.test_image_path = temp_image.name
        return temp_image.name
    
    @patch('src.ai_analyzer.safety_detector.YOLO')
    def test_safety_detector_initialization(self, mock_yolo):
        """SafetyDetector 초기화 테스트"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = SafetyDetector()
        
        self.assertIsNotNone(detector.model)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertIn('person', detector.korean_labels)
        self.assertIn('helmet', detector.korean_labels)
    
    def test_calculate_overall_risk(self):
        """전체 위험도 계산 테스트"""
        # 안전한 상황
        safe_summary = {
            'person_count': 2,
            'helmet_count': 2,
            'safety_vest_count': 2,
            'seatbelt_count': 2,
            'no_helmet_count': 0,
            'no_safety_vest_count': 0,
            'no_seatbelt_count': 0
        }
        
        risk_level = self.detector._calculate_overall_risk(safe_summary)
        self.assertEqual(risk_level, 'safe')
        
        # 위험한 상황
        risky_summary = {
            'person_count': 2,
            'helmet_count': 0,
            'safety_vest_count': 0,
            'seatbelt_count': 0,
            'no_helmet_count': 2,
            'no_safety_vest_count': 2,
            'no_seatbelt_count': 2
        }
        
        risk_level = self.detector._calculate_overall_risk(risky_summary)
        self.assertEqual(risk_level, 'high_risk')
        
        # 사람이 없는 상황
        no_person_summary = {
            'person_count': 0,
            'helmet_count': 0,
            'safety_vest_count': 0,
            'seatbelt_count': 0,
            'no_helmet_count': 0,
            'no_safety_vest_count': 0,
            'no_seatbelt_count': 0
        }
        
        risk_level = self.detector._calculate_overall_risk(no_person_summary)
        self.assertEqual(risk_level, 'no_person_detected')
    
    def test_heuristic_safety_detection(self):
        """휴리스틱 안전장비 검출 테스트"""
        image_path = self.create_test_image()
        image = cv2.imread(image_path)
        
        result = self.detector._heuristic_safety_detection(image)
        
        self.assertIn('helmet_regions', result)
        self.assertIn('vest_regions', result)
        self.assertIn('estimated_safety_score', result)
        self.assertIsInstance(result['estimated_safety_score'], float)
        self.assertGreaterEqual(result['estimated_safety_score'], 0)
        self.assertLessEqual(result['estimated_safety_score'], 1)
    
    def test_risk_to_score_conversion(self):
        """위험도-점수 변환 테스트"""
        self.assertEqual(self.detector._risk_to_score('safe'), 0)
        self.assertEqual(self.detector._risk_to_score('low_risk'), 1)
        self.assertEqual(self.detector._risk_to_score('medium_risk'), 2)
        self.assertEqual(self.detector._risk_to_score('high_risk'), 3)
        self.assertEqual(self.detector._risk_to_score('unknown'), 0)
    
    @patch('cv2.imwrite')
    def test_annotate_image(self, mock_imwrite):
        """이미지 주석 생성 테스트"""
        image_path = self.create_test_image()
        mock_imwrite.return_value = True
        
        detections = [
            {
                'class_name': 'helmet',
                'korean_label': '안전모 착용',
                'confidence': 0.95,
                'bbox': [100, 50, 150, 100],
                'risk_level': 'safe'
            },
            {
                'class_name': 'person',
                'korean_label': '작업자',
                'confidence': 0.87,
                'bbox': [80, 60, 170, 200],
                'risk_level': 'attention'
            }
        ]
        
        output_path = self.detector.annotate_image(image_path, detections)
        
        self.assertIsInstance(output_path, str)
        mock_imwrite.assert_called_once()
    
    def test_generate_safety_report(self):
        """안전 보고서 생성 테스트"""
        # 모의 분석 결과 데이터
        analysis_results = {
            'image1.jpg': {
                'status': 'success',
                'overall_risk': 'safe',
                'safety_summary': {
                    'person_count': 1,
                    'helmet_count': 1,
                    'no_helmet_count': 0,
                    'safety_vest_count': 1,
                    'no_safety_vest_count': 0,
                    'seatbelt_count': 1,
                    'no_seatbelt_count': 0
                }
            },
            'image2.jpg': {
                'status': 'success',
                'overall_risk': 'high_risk',
                'safety_summary': {
                    'person_count': 1,
                    'helmet_count': 0,
                    'no_helmet_count': 1,
                    'safety_vest_count': 0,
                    'no_safety_vest_count': 1,
                    'seatbelt_count': 0,
                    'no_seatbelt_count': 1
                }
            }
        }
        
        report = self.detector.generate_safety_report(analysis_results)
        
        # 보고서 구조 확인
        self.assertIn('summary', report)
        self.assertIn('violations', report)
        self.assertIn('recommendations', report)
        
        # 요약 정보 확인
        summary = report['summary']
        self.assertEqual(summary['total_images_analyzed'], 2)
        self.assertEqual(summary['safe_images'], 1)
        self.assertEqual(summary['risk_images'], 1)
        self.assertEqual(summary['safety_score'], 50.0)
        
        # 권고사항 확인
        self.assertIsInstance(report['recommendations'], list)
        self.assertGreater(len(report['recommendations']), 0)
    
    def test_generate_recommendations(self):
        """권고사항 생성 테스트"""
        violations = [
            {'violation': '안전모 미착용', 'count': 3, 'risk_level': 'high'},
            {'violation': '안전벨트 미착용', 'count': 2, 'risk_level': 'high'},
            {'violation': '안전조끼 미착용', 'count': 1, 'risk_level': 'medium'}
        ]
        
        recommendations = self.detector._generate_recommendations(violations)
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # 각 위반 유형에 대한 권고사항이 포함되었는지 확인
        recommendation_text = ' '.join(recommendations)
        self.assertIn('안전모', recommendation_text)
        self.assertIn('안전벨트', recommendation_text)
        self.assertIn('안전조끼', recommendation_text)
    
    def test_merge_results(self):
        """결과 통합 테스트"""
        yolo_result = {
            'detections': [{'class_name': 'person', 'confidence': 0.9}],
            'detection_count': 1
        }
        
        heuristic_result = {
            'helmet_regions': [{'bbox': [100, 50, 150, 100]}],
            'vest_regions': [],
            'estimated_safety_score': 0.6
        }
        
        merged = self.detector._merge_results(yolo_result, heuristic_result)
        
        self.assertIn('heuristic_analysis', merged)
        self.assertIn('additional_detections', merged)
        self.assertEqual(merged['detection_count'], 1)
        self.assertEqual(merged['additional_detections']['helmet_regions_found'], 1)

class TestSafetyDetectorIntegration(unittest.TestCase):
    """SafetyDetector 통합 테스트"""
    
    def setUp(self):
        """통합 테스트 설정"""
        # 실제 모델 없이 모킹
        with patch('src.ai_analyzer.safety_detector.YOLO'):
            self.detector = SafetyDetector()
    
    def test_batch_analyze_empty_list(self):
        """빈 이미지 리스트 일괄 분석 테스트"""
        results = self.detector.batch_analyze([])
        self.assertEqual(len(results), 0)
    
    def test_batch_analyze_nonexistent_files(self):
        """존재하지 않는 파일들 일괄 분석 테스트"""
        fake_paths = ['/nonexistent/image1.jpg', '/nonexistent/image2.jpg']
        results = self.detector.batch_analyze(fake_paths)
        
        self.assertEqual(len(results), 2)
        for path in fake_paths:
            self.assertIn(path, results)
            self.assertEqual(results[path]['status'], 'failed')

if __name__ == '__main__':
    unittest.main()