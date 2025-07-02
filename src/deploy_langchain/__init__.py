"""
Deploy LangChain - PDF RAG Application

PDF 파일을 업로드하고 질문하면 AI가 문서 내용을 바탕으로 답변하는 애플리케이션
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "you@example.com"

# 메인 모듈들 import
try:
    from ..rag_service import PDFRAGService
    from ..ui_interface import PDFChatInterface
    __all__ = ["PDFRAGService", "PDFChatInterface"]
except ImportError:
    # 상대 import가 실패하는 경우 (직접 실행시)
    __all__ = []