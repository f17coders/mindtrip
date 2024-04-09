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


# 마을 데이터
class Village(Base):
    __tablename__ = 'village'
    village_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    create_time = Column(DateTime, default=datetime.now(), nullable=False)
    update_time = Column(DateTime, default=datetime.now(), nullable=True)


# 결과로 도출될 수 있는 데이터
class Result(Base):
    __tablename__ = 'htp_result'
    
    result_code = Column(String, primary_key=True)
    village_id = Column(Integer, nullable=False)
    village_name = Column(String, unique=True, nullable=False)
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


class Member(Base):
    __tablename__ = "member"

    member_id = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String, default=None, unique=True)
    password = Column(String)
    social_id = Column(String)
    nickname = Column(String, unique=True)
    village_id = Column(Integer, nullable=False)
    level = Column(Integer, default=0, nullable=False)
    mission_count = Column(Integer, default=0, nullable=False)
    role = Column(SQLEnum("USER", "ADMIN"), default="USER")
    create_time = Column(DateTime, default=datetime.now(), nullable=False)
    update_time = Column(DateTime, default=datetime.now(), nullable=True)
    