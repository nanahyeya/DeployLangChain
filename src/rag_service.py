# src/rag_service.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# langchain 패키지
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")


class PDFRAGService:
    """PDF RAG 서비스 클래스"""
    
    def __init__(self):
        self.vectorstore = None
        self.current_pdf_path = None
        self.embeddings = OpenAIEmbeddings(
            api_key=OPENAI_API_KEY, 
            model="text-embedding-3-small"
        )
    
    def load_pdf_to_vectorstore(self, pdf_file_path, chunk_size=1000, chunk_overlap=200):
        """PDF 파일을 로드하고 벡터 저장소 생성"""
        try:
            print(f"PDF 파일 로딩 중: {pdf_file_path}")
            
            # PDF 파일 로딩
            loader = PyPDFLoader(pdf_file_path)
            documents = loader.load()
            
            if not documents:
                raise ValueError("PDF 파일에서 텍스트를 추출할 수 없습니다.")
            
            print(f"총 {len(documents)}페이지 로드됨")

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(chunk_size), 
                chunk_overlap=int(chunk_overlap),
                separators=["\n\n", "\n", ".", " ", ""]
            )
            splits = text_splitter.split_documents(documents)
            print(f"총 {len(splits)}개 청크로 분할됨")

            # FAISS 벡터 저장소 생성
            print("FAISS 벡터 저장소 생성 중...")
            self.vectorstore = FAISS.from_documents(
                documents=splits, 
                embedding=self.embeddings
            )
            self.current_pdf_path = pdf_file_path
            
            print("벡터 저장소 생성 완료!")
            return True
            
        except Exception as e:
            print(f"PDF 로딩 중 오류 발생: {str(e)}")
            raise e

    def get_answer(self, question, temperature=0.0):
        """질문에 대한 답변 생성"""
        if not self.vectorstore:
            return "먼저 PDF 파일을 업로드해주세요."
        
        try:
            # 검색기 설정
            retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 6}
            )

            # 프롬프트 템플릿
            template = '''다음 문맥을 바탕으로 질문에 정확하게 답변해주세요. 
문맥에서 관련 정보를 찾을 수 없다면, "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답변해주세요.

<문맥>
{context}
</문맥>

질문: {input}

답변:'''

            prompt = ChatPromptTemplate.from_template(template)
            
            # LLM 모델 설정
            model = ChatOpenAI(
                model='gpt-3.5-turbo', 
                temperature=float(temperature),
                api_key=OPENAI_API_KEY
            )

            # RAG 체인 생성
            document_chain = create_stuff_documents_chain(model, prompt)
            rag_chain = create_retrieval_chain(retriever, document_chain)
            
            # 답변 생성
            response = rag_chain.invoke({'input': question})
            return response['answer']
            
        except Exception as e:
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

    def is_pdf_loaded(self):
        """PDF가 로드되었는지 확인"""
        return self.vectorstore is not None

    def clear_vectorstore(self):
        """벡터 저장소 초기화"""
        self.vectorstore = None
        self.current_pdf_path = None
        print("벡터 저장소가 초기화되었습니다.")