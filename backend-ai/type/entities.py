from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Enum as SQLEnum, DateTime

from type.domains import DBDomain

from datetime import datetime


Base = declarative_base()

# 검사 질문
class Question(Base):
    __tablename__ = 'htp_question'
    
    question_id = Column(Integer, primary_key=True, autoincrement=True)
    domain = Column(SQLEnum(DBDomain), nullable=False)
    content = Column(String, nullable=False)

# 각 질문에 대한 선택지
class Choice(Base):
    __tablename__ = 'htp_choice'
    
    choice_id = Column(Integer, primary_key=True, autoincrement=True)
    question_id = Column(Integer, nullable=False)
    content = Column(String, nullable=False)
    score = Column(Integer, nullable=False)


# 결과로 도출될 수 있는 데이터
class Result(Base):
    __tablename__ = 'htp_result'
    
    result_code = Column(String, primary_key=True)
    village = Column(String, unique=True, nullable=False)
    content = Column(String, nullable=False)


# 실제 사용자가 검사한 결과로 도출된 데이터
class MemberResult(Base):
    __tablename__ = 'htp_member_result'
    
    member_result_id = Column(Integer, primary_key=True, autoincrement=True)
    result_code = Column(String, unique=True, nullable=False)
    member_id = Column(Integer, nullable=False)
    test_time = Column(DateTime, default=datetime.now())
    house_url = Column(String, nullable=False)
    tree_url = Column(String, nullable=False)
    person_url = Column(String, nullable=False)
    