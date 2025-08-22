"""
Flask 웹 애플리케이션 - 탱크로리 안전상태 체크 시스템
Flask web application for tanker truck safety monitoring system
"""
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import uuid
import threading
import time

# 프로젝트 모듈 import
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import Config
from src.video_processor.frame_extractor import FrameExtractor
from src.ai_analyzer.safety_detector import SafetyDetector
from src.visualizer.report_generator import SafetyReportGenerator

class SafetyVisionApp:
    """안전 비전 AI 웹 애플리케이션 클래스"""
    
    def __init__(self):
        self.app = Flask(__name__, 
                        template_folder='../../templates',
                        static_folder='../../static')
        self.app.config.from_object(Config)
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 컴포넌트 초기화
        self.frame_extractor = FrameExtractor(output_dir="output/extracted_frames")
        self.safety_detector = SafetyDetector()
        self.report_generator = SafetyReportGenerator()
        
        # 작업 상태 관리
        self.analysis_jobs = {}
        
        # 라우트 설정
        self._setup_routes()
    
    def _setup_routes(self):
        """웹 애플리케이션 라우트 설정"""
        
        @self.app.route('/')
        def index():
            """메인 페이지"""
            return render_template('index.html')
        
        @self.app.route('/upload', methods=['GET', 'POST'])
        def upload_file():
            """파일 업로드 처리"""
            if request.method == 'POST':
                if 'file' not in request.files:
                    flash('파일이 선택되지 않았습니다.')
                    return redirect(request.url)
                
                file = request.files['file']
                if file.filename == '':
                    flash('파일이 선택되지 않았습니다.')
                    return redirect(request.url)
                
                if file and self._allowed_file(file.filename):
                    # 안전한 파일명 생성
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{timestamp}_{filename}"
                    
                    # 파일 타입에 따라 저장 경로 결정
                    if self._is_video_file(filename):
                        filepath = Path(self.app.config['UPLOAD_FOLDER']) / 'videos' / unique_filename
                    else:
                        filepath = Path(self.app.config['UPLOAD_FOLDER']) / 'images' / unique_filename
                    
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    file.save(str(filepath))
                    
                    self.logger.info(f"파일 업로드 완료: {unique_filename}")
                    
                    # 분석 작업 시작
                    job_id = str(uuid.uuid4())
                    self._start_analysis_job(job_id, str(filepath), filename)
                    
                    return jsonify({
                        'success': True,
                        'job_id': job_id,
                        'filename': unique_filename,
                        'message': '파일 업로드가 완료되었습니다. 분석을 시작합니다.'
                    })
                else:
                    flash('지원하지 않는 파일 형식입니다.')
                    return redirect(request.url)
            
            return render_template('upload.html')
        
        @self.app.route('/api/job_status/<job_id>')
        def get_job_status(job_id):
            """분석 작업 상태 조회"""
            job = self.analysis_jobs.get(job_id)
            if not job:
                return jsonify({'error': '작업을 찾을 수 없습니다.'}), 404
            
            return jsonify(job)
        
        @self.app.route('/api/results/<job_id>')
        def get_analysis_results(job_id):
            """분석 결과 조회"""
            job = self.analysis_jobs.get(job_id)
            if not job:
                return jsonify({'error': '작업을 찾을 수 없습니다.'}), 404
            
            if job['status'] != 'completed':
                return jsonify({'error': '분석이 완료되지 않았습니다.'}), 400
            
            return jsonify(job['results'])
        
        @self.app.route('/results/<job_id>')
        def show_results(job_id):
            """분석 결과 페이지"""
            job = self.analysis_jobs.get(job_id)
            if not job:
                flash('작업을 찾을 수 없습니다.')
                return redirect(url_for('index'))
            
            return render_template('results.html', job_id=job_id, job=job)
        
        @self.app.route('/api/download/report/<job_id>')
        def download_report(job_id):
            """PDF 보고서 다운로드"""
            job = self.analysis_jobs.get(job_id)
            if not job or job['status'] != 'completed':
                return jsonify({'error': '보고서를 찾을 수 없습니다.'}), 404
            
            pdf_path = job.get('pdf_report_path')
            if pdf_path and Path(pdf_path).exists():
                return send_file(pdf_path, as_attachment=True, 
                               download_name=f'safety_report_{job_id[:8]}.pdf')
            else:
                return jsonify({'error': '보고서 파일을 찾을 수 없습니다.'}), 404
        
        @self.app.route('/api/download/dashboard/<job_id>')
        def download_dashboard(job_id):
            """대시보드 이미지 다운로드"""
            job = self.analysis_jobs.get(job_id)
            if not job or job['status'] != 'completed':
                return jsonify({'error': '대시보드를 찾을 수 없습니다.'}), 404
            
            dashboard_path = job.get('dashboard_path')
            if dashboard_path and Path(dashboard_path).exists():
                return send_file(dashboard_path, as_attachment=True,
                               download_name=f'safety_dashboard_{job_id[:8]}.png')
            else:
                return jsonify({'error': '대시보드 파일을 찾을 수 없습니다.'}), 404
        
        @self.app.route('/gallery/<job_id>')
        def show_gallery(job_id):
            """이미지 갤러리 페이지"""
            job = self.analysis_jobs.get(job_id)
            if not job or job['status'] != 'completed':
                flash('갤러리를 찾을 수 없습니다.')
                return redirect(url_for('index'))
            
            gallery_path = job.get('gallery_path')
            if gallery_path and Path(gallery_path).exists():
                return send_file(gallery_path)
            else:
                flash('갤러리 파일을 찾을 수 없습니다.')
                return redirect(url_for('show_results', job_id=job_id))
        
        @self.app.errorhandler(404)
        def not_found(error):
            return render_template('error.html', 
                                 error_code=404, 
                                 error_message='페이지를 찾을 수 없습니다.'), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return render_template('error.html',
                                 error_code=500,
                                 error_message='서버 내부 오류가 발생했습니다.'), 500
    
    def _allowed_file(self, filename):
        """허용된 파일 형식인지 확인"""
        if '.' not in filename:
            return False
        
        ext = '.' + filename.rsplit('.', 1)[1].lower()
        allowed_extensions = (Config.SUPPORTED_VIDEO_FORMATS + 
                            Config.SUPPORTED_IMAGE_FORMATS)
        
        return ext in allowed_extensions
    
    def _is_video_file(self, filename):
        """비디오 파일인지 확인"""
        if '.' not in filename:
            return False
        
        ext = '.' + filename.rsplit('.', 1)[1].lower()
        return ext in Config.SUPPORTED_VIDEO_FORMATS
    
    def _start_analysis_job(self, job_id: str, filepath: str, original_filename: str):
        """분석 작업을 백그라운드에서 시작"""
        
        # 작업 상태 초기화
        self.analysis_jobs[job_id] = {
            'id': job_id,
            'status': 'processing',
            'progress': 0,
            'message': '분석을 시작하고 있습니다...',
            'filepath': filepath,
            'original_filename': original_filename,
            'created_at': datetime.now().isoformat(),
            'results': None
        }
        
        # 백그라운드 스레드에서 분석 실행
        thread = threading.Thread(target=self._process_analysis, args=(job_id, filepath))
        thread.daemon = True
        thread.start()
    
    def _process_analysis(self, job_id: str, filepath: str):
        """실제 분석 처리 (백그라운드)"""
        try:
            job = self.analysis_jobs[job_id]
            
            # 1. 프레임 추출 (비디오인 경우)
            if self._is_video_file(filepath):
                job['message'] = '비디오에서 프레임을 추출하고 있습니다...'
                job['progress'] = 10
                
                extracted_frames = self.frame_extractor.extract_frames(
                    filepath, 
                    interval_seconds=Config.FRAME_EXTRACTION_INTERVAL,
                    max_frames=100  # 최대 100프레임으로 제한
                )
                
                if not extracted_frames:
                    raise Exception("프레임을 추출할 수 없습니다.")
                
                analysis_images = extracted_frames
            else:
                # 단일 이미지인 경우
                analysis_images = [filepath]
            
            job['progress'] = 30
            job['message'] = f'{len(analysis_images)}개의 이미지를 분석하고 있습니다...'
            
            # 2. AI 안전장비 검출
            analysis_results = {}
            for i, image_path in enumerate(analysis_images):
                try:
                    result = self.safety_detector.detect_safety_equipment(image_path)
                    analysis_results[image_path] = result
                    
                    # 진행률 업데이트
                    progress = 30 + (i + 1) / len(analysis_images) * 40
                    job['progress'] = int(progress)
                    
                except Exception as e:
                    self.logger.error(f"이미지 분석 실패 {image_path}: {str(e)}")
                    analysis_results[image_path] = {
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
            
            job['progress'] = 70
            job['message'] = '분석 결과를 처리하고 있습니다...'
            
            # 3. 보고서 생성
            safety_report = self.safety_detector.generate_safety_report(analysis_results)
            
            job['progress'] = 80
            job['message'] = '시각화 자료를 생성하고 있습니다...'
            
            # 4. 시각화 생성
            dashboard_path = self.report_generator.generate_visualization_dashboard(
                analysis_results,
                output_path=f"output/reports/dashboard_{job_id}.png"
            )
            
            job['progress'] = 90
            job['message'] = 'PDF 보고서를 생성하고 있습니다...'
            
            # 5. PDF 보고서 생성
            pdf_report_path = self.report_generator.generate_pdf_report(
                analysis_results,
                safety_report,
                output_path=f"output/reports/report_{job_id}.pdf"
            )
            
            # 6. 이미지 갤러리 생성
            gallery_path = self.report_generator.create_annotated_image_gallery(
                analysis_results,
                output_dir=f"output/reports/gallery_{job_id}"
            )
            
            # 7. 주석이 추가된 이미지 생성
            annotated_images = []
            for image_path, result in analysis_results.items():
                if result.get('status') == 'success' and result.get('detections'):
                    try:
                        annotated_path = self.safety_detector.annotate_image(
                            image_path,
                            result['detections'],
                            output_path=f"output/annotated_images/annotated_{Path(image_path).name}"
                        )
                        annotated_images.append(annotated_path)
                    except Exception as e:
                        self.logger.error(f"이미지 주석 생성 실패 {image_path}: {str(e)}")
            
            # 작업 완료
            job['status'] = 'completed'
            job['progress'] = 100
            job['message'] = '분석이 완료되었습니다.'
            job['completed_at'] = datetime.now().isoformat()
            job['results'] = {
                'analysis_results': analysis_results,
                'safety_report': safety_report,
                'annotated_images': annotated_images,
                'total_images': len(analysis_images),
                'safe_images': safety_report['summary']['safe_images'],
                'risk_images': safety_report['summary']['risk_images'],
                'safety_score': safety_report['summary']['safety_score']
            }
            job['dashboard_path'] = dashboard_path
            job['pdf_report_path'] = pdf_report_path
            job['gallery_path'] = gallery_path
            
            self.logger.info(f"분석 작업 완료: {job_id}")
            
        except Exception as e:
            # 오류 처리
            self.logger.error(f"분석 작업 실패 {job_id}: {str(e)}")
            job['status'] = 'failed'
            job['message'] = f'분석 중 오류가 발생했습니다: {str(e)}'
            job['error'] = str(e)
            job['failed_at'] = datetime.now().isoformat()
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """웹 애플리케이션 실행"""
        self.logger.info(f"SafetyVision AI 웹 애플리케이션 시작: http://{host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_app():
    """Flask 애플리케이션 팩토리"""
    app_instance = SafetyVisionApp()
    return app_instance.app

if __name__ == '__main__':
    app = SafetyVisionApp()
    app.run(debug=True)