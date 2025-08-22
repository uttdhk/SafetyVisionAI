"""
안전장비 검출을 위한 AI 분석 모듈
AI analysis module for safety equipment detection
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

class SafetyDetector:
    """안전장비 검출을 위한 AI 모델 클래스"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Args:
            model_path: 커스텀 모델 경로 (None이면 기본 YOLO 모델 사용)
            confidence_threshold: 검출 신뢰도 임계값
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        
        # YOLO 모델 로드
        try:
            if model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                self.logger.info(f"커스텀 모델 로드: {model_path}")
            else:
                # 기본 YOLOv8 모델 사용 (person 검출용)
                self.model = YOLO('yolov8n.pt')
                self.logger.info("기본 YOLOv8 모델 로드")
                
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {str(e)}")
            raise
        
        # 안전장비 클래스 정의
        self.safety_classes = {
            # YOLO 기본 클래스
            0: 'person',
            
            # 커스텀 안전장비 클래스 (실제 훈련된 모델에 따라 조정)
            80: 'helmet',
            81: 'no_helmet', 
            82: 'safety_vest',
            83: 'no_safety_vest',
            84: 'seatbelt',
            85: 'no_seatbelt'
        }
        
        # 한국어 라벨
        self.korean_labels = {
            'person': '작업자',
            'helmet': '안전모 착용',
            'no_helmet': '안전모 미착용',
            'safety_vest': '안전조끼 착용',
            'no_safety_vest': '안전조끼 미착용',
            'seatbelt': '안전벨트 착용',
            'no_seatbelt': '안전벨트 미착용'
        }
        
        # 위험도 레벨 정의
        self.risk_levels = {
            'helmet': 'safe',
            'no_helmet': 'high_risk',
            'safety_vest': 'safe',
            'no_safety_vest': 'medium_risk',
            'seatbelt': 'safe',
            'no_seatbelt': 'high_risk',
            'person': 'attention'
        }
        
        # 색상 정의 (BGR 형식)
        self.colors = {
            'safe': (0, 255, 0),        # 녹색
            'medium_risk': (0, 165, 255), # 주황색
            'high_risk': (0, 0, 255),   # 빨간색
            'attention': (255, 255, 0)   # 노란색
        }
    
    def detect_safety_equipment(self, image_path: str) -> Dict:
        """
        이미지에서 안전장비를 검출하고 분석 결과를 반환
        
        Args:
            image_path: 분석할 이미지 경로
            
        Returns:
            검출 결과 딕셔너리
        """
        if not Path(image_path).exists():
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # YOLO 검출 수행
        results = self.model(image, conf=self.confidence_threshold)
        
        # 결과 분석
        analysis_result = self._analyze_detections(results, image_path)
        
        # 추가 휴리스틱 분석 (색상 기반 안전장비 검출)
        heuristic_result = self._heuristic_safety_detection(image)
        
        # 결과 통합
        final_result = self._merge_results(analysis_result, heuristic_result)
        
        return final_result
    
    def batch_analyze(self, image_paths: List[str]) -> Dict[str, Dict]:
        """여러 이미지를 일괄 분석"""
        results = {}
        
        for image_path in image_paths:
            try:
                result = self.detect_safety_equipment(image_path)
                results[image_path] = result
                self.logger.info(f"분석 완료: {Path(image_path).name}")
                
            except Exception as e:
                self.logger.error(f"분석 실패 {image_path}: {str(e)}")
                results[image_path] = {
                    'status': 'failed',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return results
    
    def _analyze_detections(self, results, image_path: str) -> Dict:
        """YOLO 검출 결과를 분석"""
        detections = []
        safety_summary = {
            'helmet_count': 0,
            'no_helmet_count': 0,
            'safety_vest_count': 0,
            'no_safety_vest_count': 0,
            'seatbelt_count': 0,
            'no_seatbelt_count': 0,
            'person_count': 0
        }
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 클래스 ID와 신뢰도 추출
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 클래스명 매핑
                    class_name = self.safety_classes.get(class_id, f'unknown_{class_id}')
                    
                    detection = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'korean_label': self.korean_labels.get(class_name, class_name),
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'risk_level': self.risk_levels.get(class_name, 'unknown')
                    }
                    
                    detections.append(detection)
                    
                    # 통계 업데이트
                    if class_name in safety_summary:
                        if class_name == 'person':
                            safety_summary['person_count'] += 1
                        else:
                            safety_summary[f'{class_name}_count'] += 1
        
        # 전체 위험도 평가
        overall_risk = self._calculate_overall_risk(safety_summary)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'safety_summary': safety_summary,
            'overall_risk': overall_risk,
            'detection_count': len(detections),
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
    
    def _heuristic_safety_detection(self, image: np.ndarray) -> Dict:
        """색상과 형태 기반 휴리스틱 안전장비 검출"""
        heuristic_detections = {
            'helmet_regions': [],
            'vest_regions': [],
            'estimated_safety_score': 0.0
        }
        
        # HSV 색공간으로 변환
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        
        # 안전모 색상 범위 (일반적으로 노란색, 흰색, 빨간색)
        helmet_colors = [
            # 노란색 안전모
            (np.array([20, 100, 100]), np.array([30, 255, 255])),
            # 흰색 안전모  
            (np.array([0, 0, 200]), np.array([180, 30, 255])),
            # 빨간색 안전모
            (np.array([0, 100, 100]), np.array([10, 255, 255]))
        ]
        
        # 안전조끼 색상 범위 (주로 형광 노란색, 주황색)
        vest_colors = [
            # 형광 노란색
            (np.array([25, 100, 200]), np.array([35, 255, 255])),
            # 주황색
            (np.array([5, 100, 100]), np.array([15, 255, 255]))
        ]
        
        # 안전모 검출
        for color_range in helmet_colors:
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
            
            # 상반신 영역에서만 검색 (이미지 상단 60%)
            mask_roi = mask[:int(height * 0.6), :]
            
            # 형태학적 연산으로 노이즈 제거
            kernel = np.ones((5, 5), np.uint8)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # 적절한 크기의 영역만 선택
                if 500 < area < 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # 종횡비 확인 (안전모는 대략 정사각형에 가까움)
                    if 0.7 < w/h < 1.5:
                        heuristic_detections['helmet_regions'].append({
                            'bbox': [x, y, x+w, y+h],
                            'area': area,
                            'confidence': min(area / 2000, 1.0)
                        })
        
        # 안전조끼 검출
        for color_range in vest_colors:
            mask = cv2.inRange(hsv, color_range[0], color_range[1])
            
            # 중간 부분 영역에서 검색 (이미지 중앙 60%)
            roi_start = int(height * 0.2)
            roi_end = int(height * 0.8)
            mask_roi = mask[roi_start:roi_end, :]
            
            # 형태학적 연산
            kernel = np.ones((7, 7), np.uint8)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel)
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # 안전조끼는 더 큰 영역
                if 2000 < area < 20000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # 안전조끼는 세로가 더 김
                    if h > w and h/w > 1.2:
                        heuristic_detections['vest_regions'].append({
                            'bbox': [x, y+roi_start, x+w, y+h+roi_start],
                            'area': area,
                            'confidence': min(area / 10000, 1.0)
                        })
        
        # 전체 안전 점수 계산
        helmet_score = len(heuristic_detections['helmet_regions']) * 0.4
        vest_score = len(heuristic_detections['vest_regions']) * 0.3
        heuristic_detections['estimated_safety_score'] = min(helmet_score + vest_score, 1.0)
        
        return heuristic_detections
    
    def _merge_results(self, yolo_result: Dict, heuristic_result: Dict) -> Dict:
        """YOLO 결과와 휴리스틱 결과를 통합"""
        merged_result = yolo_result.copy()
        merged_result['heuristic_analysis'] = heuristic_result
        
        # 휴리스틱으로 발견된 안전장비 정보 추가
        merged_result['additional_detections'] = {
            'helmet_regions_found': len(heuristic_result['helmet_regions']),
            'vest_regions_found': len(heuristic_result['vest_regions']),
            'estimated_safety_score': heuristic_result['estimated_safety_score']
        }
        
        return merged_result
    
    def _calculate_overall_risk(self, safety_summary: Dict) -> str:
        """전체 위험도를 계산"""
        total_persons = safety_summary['person_count']
        
        if total_persons == 0:
            return 'no_person_detected'
        
        # 위험 요소 점수 계산
        risk_score = 0
        
        # 안전모 미착용
        if safety_summary['no_helmet_count'] > 0:
            risk_score += safety_summary['no_helmet_count'] * 3
        
        # 안전벨트 미착용 (탱크로리에서 중요)
        if safety_summary['no_seatbelt_count'] > 0:
            risk_score += safety_summary['no_seatbelt_count'] * 3
        
        # 안전조끼 미착용
        if safety_summary['no_safety_vest_count'] > 0:
            risk_score += safety_summary['no_safety_vest_count'] * 2
        
        # 위험도 레벨 결정
        if risk_score == 0:
            return 'safe'
        elif risk_score <= 2:
            return 'low_risk'
        elif risk_score <= 5:
            return 'medium_risk'
        else:
            return 'high_risk'
    
    def annotate_image(self, image_path: str, detections: List[Dict], 
                      output_path: Optional[str] = None) -> str:
        """
        검출 결과로 이미지에 주석을 추가
        
        Args:
            image_path: 원본 이미지 경로
            detections: 검출 결과 리스트
            output_path: 출력 이미지 경로 (None이면 자동 생성)
            
        Returns:
            주석이 추가된 이미지 경로
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # 각 검출 결과에 대해 바운딩 박스와 라벨 추가
        for detection in detections:
            bbox = detection['bbox']
            class_name = detection['class_name']
            confidence = detection['confidence']
            risk_level = detection['risk_level']
            korean_label = detection['korean_label']
            
            # 위험도에 따른 색상 선택
            color = self.colors.get(risk_level, (128, 128, 128))
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # 라벨 텍스트 준비
            label = f"{korean_label} ({confidence:.2f})"
            
            # 텍스트 배경 그리기
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1] - 10),
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # 텍스트 그리기
            cv2.putText(image, label, (bbox[0], bbox[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 출력 경로 설정
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"annotated_{input_path.name}"
        
        # 주석이 추가된 이미지 저장
        cv2.imwrite(str(output_path), image)
        
        return str(output_path)
    
    def generate_safety_report(self, analysis_results: Dict) -> Dict:
        """분석 결과를 바탕으로 안전 보고서를 생성"""
        total_images = len(analysis_results)
        safe_images = 0
        risk_images = 0
        violations = []
        
        for image_path, result in analysis_results.items():
            if result.get('status') != 'success':
                continue
                
            overall_risk = result.get('overall_risk', 'unknown')
            
            if overall_risk == 'safe':
                safe_images += 1
            else:
                risk_images += 1
                
                # 위반 사항 수집
                safety_summary = result.get('safety_summary', {})
                if safety_summary.get('no_helmet_count', 0) > 0:
                    violations.append({
                        'image': Path(image_path).name,
                        'violation': '안전모 미착용',
                        'count': safety_summary['no_helmet_count'],
                        'risk_level': 'high'
                    })
                
                if safety_summary.get('no_seatbelt_count', 0) > 0:
                    violations.append({
                        'image': Path(image_path).name,
                        'violation': '안전벨트 미착용',
                        'count': safety_summary['no_seatbelt_count'],
                        'risk_level': 'high'
                    })
                
                if safety_summary.get('no_safety_vest_count', 0) > 0:
                    violations.append({
                        'image': Path(image_path).name,
                        'violation': '안전조끼 미착용',
                        'count': safety_summary['no_safety_vest_count'],
                        'risk_level': 'medium'
                    })
        
        # 안전도 점수 계산
        safety_score = (safe_images / total_images * 100) if total_images > 0 else 0
        
        report = {
            'summary': {
                'total_images_analyzed': total_images,
                'safe_images': safe_images,
                'risk_images': risk_images,
                'safety_score': round(safety_score, 2),
                'total_violations': len(violations)
            },
            'violations': violations,
            'recommendations': self._generate_recommendations(violations),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, violations: List[Dict]) -> List[str]:
        """위반 사항을 바탕으로 권고사항을 생성"""
        recommendations = []
        
        # 위반 유형별 카운트
        violation_counts = {}
        for violation in violations:
            v_type = violation['violation']
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
        
        if violation_counts.get('안전모 미착용', 0) > 0:
            recommendations.append("모든 작업자는 작업 전 안전모 착용을 확인하시기 바랍니다.")
            
        if violation_counts.get('안전벨트 미착용', 0) > 0:
            recommendations.append("탱크로리 운전 시 반드시 안전벨트를 착용하시기 바랍니다.")
            
        if violation_counts.get('안전조끼 미착용', 0) > 0:
            recommendations.append("시인성 향상을 위해 안전조끼 착용을 권장합니다.")
        
        if len(recommendations) == 0:
            recommendations.append("현재 안전수칙이 잘 준수되고 있습니다. 지속적인 관리를 부탁드립니다.")
        
        return recommendations