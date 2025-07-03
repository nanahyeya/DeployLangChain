# src/main.py
import os
import sys
import socket
from pathlib import Path

# 현재 파일의 디렉토리를 Python 경로에 추가 (같은 디렉토리의 모듈 import를 위해)
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from ui_interface import PDFChatInterface

def detect_docker_environment():
    """Docker 환경 감지"""
    try:
        hostname = socket.gethostname()
        is_docker = os.path.exists('/.dockerenv') or hostname.startswith('pdf-rag-container')
        return is_docker
    except:
        return False


def main():
    """메인 함수"""
    print(" PDF RAG 애플리케이션을 시작합니다...")
    
    # UI 인터페이스 생성
    chat_interface = PDFChatInterface()
    demo = chat_interface.create_interface()
    
    # Docker 환경에 따라 서버 설정 변경
    if detect_docker_environment():
        server_name = "0.0.0.0"  # Docker 컨테이너에서는 모든 인터페이스에서 접근 가능
        print(" Docker 환경에서 실행 중...")
    else:
        server_name = "127.0.0.1"  # 로컬 개발 환경
        print(" 로컬 환경에서 실행 중...")
    
    print(f" 서버 주소: {server_name}:7860")
    
    # 애플리케이션 시작
    demo.launch(
        share=False,
        debug=True,
        server_name=server_name,
        server_port=7860
    )


if __name__ == "__main__":
    main()