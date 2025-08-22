"""
안전 분석 결과 시각화 및 리포트 생성 모듈
Visualization and report generation module for safety analysis results
"""
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.units import inch
import logging
import cv2

class SafetyReportGenerator:
    """안전 분석 결과 시각화 및 PDF 리포트 생성 클래스"""
    
    def __init__(self, output_dir: str = "output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # 한글 폰트 설정 (시스템에 설치된 폰트 사용)
        self._setup_korean_font()
        
        # 색상 팔레트 설정
        self.colors = {
            'safe': '#2ECC71',      # 녹색
            'low_risk': '#F39C12',  # 주황색
            'medium_risk': '#E67E22', # 진한 주황색
            'high_risk': '#E74C3C', # 빨간색
            'no_person': '#95A5A6'  # 회색
        }
        
        # 한글 라벨
        self.korean_labels = {
            'safe': '안전',
            'low_risk': '낮은 위험',
            'medium_risk': '중간 위험', 
            'high_risk': '높은 위험',
            'no_person_detected': '작업자 없음'
        }
    
    def _setup_korean_font(self):
        """한글 폰트 설정"""
        try:
            # 시스템에서 한글 폰트 찾기
            korean_fonts = []
            for font in fm.findSystemFonts():
                try:
                    font_prop = fm.FontProperties(fname=font)
                    font_name = font_prop.get_name()
                    if any(keyword in font_name.lower() for keyword in ['nanum', 'malgun', 'dotum', 'gulim']):
                        korean_fonts.append(font)
                except:
                    continue
            
            if korean_fonts:
                plt.rcParams['font.family'] = fm.FontProperties(fname=korean_fonts[0]).get_name()
                self.korean_font = korean_fonts[0]
            else:
                # 기본 폰트 사용
                plt.rcParams['font.family'] = 'DejaVu Sans'
                self.korean_font = None
                
        except Exception as e:
            self.logger.warning(f"한글 폰트 설정 실패: {str(e)}")
            plt.rcParams['font.family'] = 'DejaVu Sans'
            self.korean_font = None
    
    def generate_visualization_dashboard(self, analysis_results: Dict, 
                                       output_path: Optional[str] = None) -> str:
        """분석 결과를 종합한 시각화 대시보드 생성"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"safety_dashboard_{timestamp}.png"
        
        # 데이터 준비
        dashboard_data = self._prepare_dashboard_data(analysis_results)
        
        # 대시보드 레이아웃 설정
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('탱크로리 안전상태 분석 대시보드', fontsize=20, fontweight='bold')
        
        # 1. 전체 안전도 점수
        self._plot_safety_score(axes[0, 0], dashboard_data)
        
        # 2. 위험도별 이미지 분포
        self._plot_risk_distribution(axes[0, 1], dashboard_data)
        
        # 3. 안전장비 착용률
        self._plot_equipment_compliance(axes[0, 2], dashboard_data)
        
        # 4. 시간별 위험도 변화 (프레임 순서 기준)
        self._plot_risk_timeline(axes[1, 0], dashboard_data)
        
        # 5. 위반 사항 상위 항목
        self._plot_violation_ranking(axes[1, 1], dashboard_data)
        
        # 6. 안전 권고사항
        self._plot_recommendations(axes[1, 2], dashboard_data)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"대시보드 생성 완료: {output_path}")
        return str(output_path)
    
    def _prepare_dashboard_data(self, analysis_results: Dict) -> Dict:
        """대시보드용 데이터 준비"""
        data = {
            'total_images': 0,
            'risk_counts': {'safe': 0, 'low_risk': 0, 'medium_risk': 0, 'high_risk': 0, 'no_person': 0},
            'equipment_stats': {
                'helmet_compliance': 0,
                'vest_compliance': 0,
                'seatbelt_compliance': 0
            },
            'violations': [],
            'timeline_data': [],
            'safety_scores': []
        }
        
        for i, (image_path, result) in enumerate(analysis_results.items()):
            if result.get('status') != 'success':
                continue
                
            data['total_images'] += 1
            
            # 위험도 통계
            risk_level = result.get('overall_risk', 'no_person')
            if risk_level == 'no_person_detected':
                risk_level = 'no_person'
            data['risk_counts'][risk_level] = data['risk_counts'].get(risk_level, 0) + 1
            
            # 안전장비 통계
            safety_summary = result.get('safety_summary', {})
            total_persons = safety_summary.get('person_count', 0)
            
            if total_persons > 0:
                helmet_safe = safety_summary.get('helmet_count', 0)
                vest_safe = safety_summary.get('safety_vest_count', 0)
                seatbelt_safe = safety_summary.get('seatbelt_count', 0)
                
                data['equipment_stats']['helmet_compliance'] += helmet_safe / total_persons
                data['equipment_stats']['vest_compliance'] += vest_safe / total_persons  
                data['equipment_stats']['seatbelt_compliance'] += seatbelt_safe / total_persons
            
            # 타임라인 데이터
            data['timeline_data'].append({
                'frame': i,
                'risk_level': risk_level,
                'risk_score': self._risk_to_score(risk_level)
            })
            
            # 개별 안전 점수 계산
            frame_score = self._calculate_frame_safety_score(result)
            data['safety_scores'].append(frame_score)
        
        # 평균 계산
        if data['total_images'] > 0:
            for key in data['equipment_stats']:
                data['equipment_stats'][key] = (data['equipment_stats'][key] / data['total_images']) * 100
        
        data['overall_safety_score'] = np.mean(data['safety_scores']) if data['safety_scores'] else 0
        
        return data
    
    def _plot_safety_score(self, ax, data: Dict):
        """전체 안전도 점수 시각화"""
        score = data['overall_safety_score']
        
        # 도넛 차트로 점수 표시
        sizes = [score, 100 - score]
        colors_list = [self._get_score_color(score), '#E5E5E5']
        
        wedges, texts = ax.pie(sizes, colors=colors_list, startangle=90, 
                              counterclock=False, wedgeprops=dict(width=0.5))
        
        # 중앙에 점수 표시
        ax.text(0, 0, f'{score:.1f}%', ha='center', va='center', 
                fontsize=24, fontweight='bold')
        ax.text(0, -0.3, '전체 안전도', ha='center', va='center', fontsize=12)
        
        ax.set_title('안전도 점수', fontsize=14, pad=20)
    
    def _plot_risk_distribution(self, ax, data: Dict):
        """위험도별 분포 차트"""
        risk_levels = list(data['risk_counts'].keys())
        counts = list(data['risk_counts'].values())
        colors_list = [self.colors.get(level, '#95A5A6') for level in risk_levels]
        labels = [self.korean_labels.get(level, level) for level in risk_levels]
        
        bars = ax.bar(labels, counts, color=colors_list)
        
        # 막대 위에 수치 표시
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('위험도별 프레임 분포', fontsize=14)
        ax.set_ylabel('프레임 수')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    def _plot_equipment_compliance(self, ax, data: Dict):
        """안전장비 착용률 차트"""
        equipment_names = ['안전모', '안전조끼', '안전벨트']
        compliance_rates = [
            data['equipment_stats']['helmet_compliance'],
            data['equipment_stats']['vest_compliance'],
            data['equipment_stats']['seatbelt_compliance']
        ]
        
        colors_list = ['#3498DB', '#9B59B6', '#1ABC9C']
        bars = ax.barh(equipment_names, compliance_rates, color=colors_list)
        
        # 막대 끝에 퍼센트 표시
        for bar, rate in zip(bars, compliance_rates):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{rate:.1f}%', ha='left', va='center', fontsize=10)
        
        ax.set_title('안전장비 착용률', fontsize=14)
        ax.set_xlabel('착용률 (%)')
        ax.set_xlim(0, 105)
    
    def _plot_risk_timeline(self, ax, data: Dict):
        """시간별 위험도 변화"""
        timeline_data = data['timeline_data']
        
        if not timeline_data:
            ax.text(0.5, 0.5, '데이터 없음', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('시간별 위험도 변화', fontsize=14)
            return
        
        frames = [item['frame'] for item in timeline_data]
        risk_scores = [item['risk_score'] for item in timeline_data]
        
        # 위험도에 따른 색상 설정
        colors_list = []
        for item in timeline_data:
            risk_level = item['risk_level']
            colors_list.append(self.colors.get(risk_level, '#95A5A6'))
        
        ax.scatter(frames, risk_scores, c=colors_list, alpha=0.7, s=50)
        ax.plot(frames, risk_scores, alpha=0.5, color='gray', linewidth=1)
        
        ax.set_title('시간별 위험도 변화', fontsize=14)
        ax.set_xlabel('프레임 순서')
        ax.set_ylabel('위험도 점수')
        ax.set_ylim(0, 4)
    
    def _plot_violation_ranking(self, ax, data: Dict):
        """위반 사항 상위 항목"""
        # 샘플 위반 사항 (실제로는 analysis_results에서 추출)
        violations = ['안전모 미착용', '안전벨트 미착용', '안전조끼 미착용']
        counts = [15, 8, 5]  # 실제로는 데이터에서 계산
        
        colors_list = ['#E74C3C', '#E67E22', '#F39C12']
        bars = ax.barh(violations, counts, color=colors_list)
        
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{count}건', ha='left', va='center', fontsize=10)
        
        ax.set_title('주요 위반 사항', fontsize=14)
        ax.set_xlabel('발생 건수')
    
    def _plot_recommendations(self, ax, data: Dict):
        """안전 권고사항"""
        ax.axis('off')
        
        recommendations = [
            "• 모든 작업자 안전모 착용 확인",
            "• 탱크로리 운전 시 안전벨트 필수",
            "• 정기적인 안전교육 실시",
            "• CCTV 모니터링 강화"
        ]
        
        ax.text(0.05, 0.95, '안전 권고사항', fontsize=14, fontweight='bold',
               transform=ax.transAxes, verticalalignment='top')
        
        for i, rec in enumerate(recommendations):
            ax.text(0.05, 0.80 - i*0.15, rec, fontsize=11,
                   transform=ax.transAxes, verticalalignment='top')
    
    def generate_pdf_report(self, analysis_results: Dict, report_data: Dict,
                           output_path: Optional[str] = None) -> str:
        """PDF 형태의 상세 안전 보고서 생성"""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"safety_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=1,  # 중앙 정렬
            spaceAfter=30
        )
        
        # 본문 스타일
        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=12
        )
        
        # 1. 보고서 제목
        story.append(Paragraph("탱크로리 안전상태 분석 보고서", title_style))
        story.append(Spacer(1, 12))
        
        # 2. 보고서 정보
        report_info = [
            ['생성일시', datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")],
            ['분석 이미지 수', str(report_data['summary']['total_images_analyzed'])],
            ['전체 안전도 점수', f"{report_data['summary']['safety_score']:.1f}%"],
            ['총 위반 건수', str(report_data['summary']['total_violations'])]
        ]
        
        info_table = Table(report_info, colWidths=[2*inch, 3*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(info_table)
        story.append(Spacer(1, 20))
        
        # 3. 분석 요약
        story.append(Paragraph("분석 요약", styles['Heading2']))
        
        summary_text = f"""
        총 {report_data['summary']['total_images_analyzed']}개의 프레임을 분석한 결과, 
        안전한 상태의 프레임이 {report_data['summary']['safe_images']}개, 
        위험 요소가 있는 프레임이 {report_data['summary']['risk_images']}개 발견되었습니다.
        전체 안전도 점수는 {report_data['summary']['safety_score']:.1f}%입니다.
        """
        
        story.append(Paragraph(summary_text, content_style))
        story.append(Spacer(1, 15))
        
        # 4. 위반사항 상세
        if report_data['violations']:
            story.append(Paragraph("발견된 위반사항", styles['Heading2']))
            
            violation_data = [['이미지', '위반 내용', '발생 건수', '위험도']]
            for violation in report_data['violations'][:10]:  # 상위 10개만 표시
                violation_data.append([
                    violation['image'],
                    violation['violation'],
                    str(violation['count']),
                    violation['risk_level']
                ])
            
            violation_table = Table(violation_data, colWidths=[1.5*inch, 2*inch, 1*inch, 1*inch])
            violation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(violation_table)
            story.append(Spacer(1, 15))
        
        # 5. 권고사항
        story.append(Paragraph("안전 권고사항", styles['Heading2']))
        
        for i, recommendation in enumerate(report_data['recommendations'], 1):
            story.append(Paragraph(f"{i}. {recommendation}", content_style))
        
        # PDF 생성
        doc.build(story)
        
        self.logger.info(f"PDF 보고서 생성 완료: {output_path}")
        return str(output_path)
    
    def create_annotated_image_gallery(self, analysis_results: Dict, 
                                     output_dir: Optional[str] = None) -> str:
        """주석이 추가된 이미지들로 갤러리 HTML 페이지 생성"""
        
        if output_dir is None:
            output_dir = self.output_dir / "gallery"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # HTML 템플릿
        html_template = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>안전 분석 결과 갤러리</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
                .image-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9; }}
                .image-card img {{ width: 100%; height: 200px; object-fit: cover; border-radius: 4px; }}
                .risk-high {{ border-left: 5px solid #E74C3C; }}
                .risk-medium {{ border-left: 5px solid #E67E22; }}
                .risk-low {{ border-left: 5px solid #F39C12; }}
                .risk-safe {{ border-left: 5px solid #2ECC71; }}
                .image-info {{ margin-top: 10px; }}
                .risk-badge {{ padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; }}
                .badge-high {{ background-color: #E74C3C; }}
                .badge-medium {{ background-color: #E67E22; }}
                .badge-low {{ background-color: #F39C12; }}
                .badge-safe {{ background-color: #2ECC71; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>탱크로리 안전상태 분석 결과</h1>
                <p>생성일시: {timestamp}</p>
            </div>
            <div class="gallery">
                {gallery_content}
            </div>
        </body>
        </html>
        """
        
        gallery_items = []
        
        for image_path, result in analysis_results.items():
            if result.get('status') != 'success':
                continue
            
            image_name = Path(image_path).name
            risk_level = result.get('overall_risk', 'unknown')
            detections = result.get('detections', [])
            
            # 위험도에 따른 CSS 클래스
            risk_class = f"risk-{risk_level.replace('_risk', '').replace('_', '-')}"
            badge_class = f"badge-{risk_level.replace('_risk', '').replace('_', '-')}"
            
            # 검출 결과 요약
            detection_summary = []
            for detection in detections[:3]:  # 상위 3개만 표시
                detection_summary.append(f"• {detection['korean_label']} ({detection['confidence']:.2f})")
            
            gallery_item = f"""
            <div class="image-card {risk_class}">
                <img src="../{image_path}" alt="{image_name}">
                <div class="image-info">
                    <h3>{image_name}</h3>
                    <span class="risk-badge {badge_class}">{self.korean_labels.get(risk_level, risk_level)}</span>
                    <div style="margin-top: 10px;">
                        <strong>검출 결과:</strong><br>
                        {'<br>'.join(detection_summary) if detection_summary else '검출된 객체 없음'}
                    </div>
                </div>
            </div>
            """
            gallery_items.append(gallery_item)
        
        # HTML 파일 생성
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S"),
            gallery_content='\n'.join(gallery_items)
        )
        
        html_path = output_dir / "safety_gallery.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"이미지 갤러리 생성 완료: {html_path}")
        return str(html_path)
    
    def _risk_to_score(self, risk_level: str) -> float:
        """위험도를 점수로 변환"""
        score_map = {
            'safe': 0,
            'low_risk': 1,
            'medium_risk': 2,
            'high_risk': 3,
            'no_person': 0
        }
        return score_map.get(risk_level, 0)
    
    def _get_score_color(self, score: float) -> str:
        """점수에 따른 색상 반환"""
        if score >= 80:
            return self.colors['safe']
        elif score >= 60:
            return self.colors['low_risk']
        elif score >= 40:
            return self.colors['medium_risk']
        else:
            return self.colors['high_risk']
    
    def _calculate_frame_safety_score(self, result: Dict) -> float:
        """개별 프레임의 안전 점수 계산"""
        if result.get('status') != 'success':
            return 0
        
        safety_summary = result.get('safety_summary', {})
        total_persons = safety_summary.get('person_count', 0)
        
        if total_persons == 0:
            return 50  # 작업자가 없는 경우 중간 점수
        
        # 안전장비 착용률 기반 점수 계산
        safe_count = (
            safety_summary.get('helmet_count', 0) +
            safety_summary.get('safety_vest_count', 0) +
            safety_summary.get('seatbelt_count', 0)
        )
        
        unsafe_count = (
            safety_summary.get('no_helmet_count', 0) +
            safety_summary.get('no_safety_vest_count', 0) +
            safety_summary.get('no_seatbelt_count', 0)
        )
        
        if safe_count + unsafe_count == 0:
            return 50  # 안전장비 정보가 없는 경우
        
        safety_ratio = safe_count / (safe_count + unsafe_count)
        return safety_ratio * 100