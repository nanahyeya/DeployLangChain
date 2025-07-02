# Python 3.12 slim 이미지 사용
FROM python:3.12-slim

# 컨테이너 내부의 작업 디렉토리를 설정
WORKDIR /app

# 시스템 의존성 설치 
# build-essential: C/C++ 컴파일러와 개발 도구들 (일부 Python 패키지 컴파일에 필요)
# curl: HTTP 요청을 위한 도구, git: 버전 관리 시스템
# 설치 후 패키지 목록 캐시를 삭제하여 이미지 크기를 최적화합니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 출력 버퍼링을 비활성화하여 실시간으로 로그를 확인할 수 있도록 함
ENV PYTHONUNBUFFERED=1
# Python이 모듈을 찾을 경로를 설정합니다.
ENV PYTHONPATH=/app

# pyproject.toml 복사 및 의존성 설치
COPY pyproject.toml ./

# pip로 의존성 직접 설치 (pyproject.toml의 dependencies 사용)
RUN pip install --no-cache-dir \
    "python-dotenv>=1.1.1,<2.0.0" \
    "langchain>=0.3.26,<0.4.0" \
    "langchain-openai>=0.3.25,<0.4.0" \
    "langchain-community>=0.3.26,<0.4.0" \
    "pypdf>=5.6.1,<6.0.0" \
    "gradio>=5.34.2,<6.0.0" \
    "gradio-pdf>=0.0.22,<0.0.23" \
    "faiss-cpu>=1.11.0,<2.0.0"

# 소스 코드 복사
COPY src/ ./src/

# 업로드된 PDF 파일을 저장할 디렉토리 생성
RUN mkdir -p /app/uploads

# Gradio가 사용하는 포트 노출
EXPOSE 7860

# 보안을 위해 root가 아닌 일반 사용자(appuser)를 생성하고 실행합니다.
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
# 애플리케이션 디렉토리의 소유권을 해당 사용자에게 부여합니다.    
USER appuser

# 애플리케이션 실행 (모듈로 실행)
CMD ["python", "src/main.py"]