# For type hints
from typing import List, Optional
from type import entities
from sqlalchemy.orm.query import Query

from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv
import os 




class MariaDBUtil():
    load_dotenv()
    __username = os.getenv("MARIADB_USERNAME")
    __password = os.getenv("MARIADB_PASSWORD")
    __host = os.getenv("MARIADB_HOST")
    __port = int(os.getenv("MARIADB_PORT_INT"))
    __database = os.getenv("MARIADB_DATABASE_NAME")
    
    
    Session = sessionmaker(bind=create_engine(f"mysql+mysqldb://{__username}:{__password}@{__host}:{__port}/{__database}"))
    
    @classmethod
    def get_questions(cls, domain) -> Optional[List[entities.Question]]:
        result = None

        with cls.Session() as session:
            query: Query[entities.Question] = session.query(entities.Question).filter_by(domain=domain)
            result = query.all()

        return result


    @classmethod
    def get_choices(cls, question_id: int) -> Optional[List[entities.Choice]]:
        result = None 

        with cls.Session() as session:
            query: Query[entities.Choice] = session.query(entities.Choice).filter_by(question_id=question_id)
            result = query.all()

        return result


    @classmethod
    def get_score(cls, choice_id: int) -> Optional[int]:
        with cls.Session() as session:
            query: Query[entities.Choice] = session.query(entities.Choice).filter_by(choice_id=choice_id)

            result = query.first()
            
            return int(result.score) if result else None


    @classmethod
    def get_result_sentence(cls, result_code: str) -> Optional[str]:
        print(f"이 코드로 결과 문장을 찾는다이: {result_code}")
        with cls.Session() as session:
            query: Query[entities.Result] = session.query(entities.Result).filter_by(result_code=result_code)

            result = query.first()

            return result.content if result else None


    @classmethod
    def get_result_village(cls, result_code: str) -> Optional[str]:
        with cls.Session() as session:
            query: Query[entities.Result] = session.query(entities.Result).filter_by(result_code=result_code)
            result = query.first()

            return result.village if result else None

    
    @classmethod
    def insert_member_result(cls, entity: entities.MemberResult) -> str:
        with cls.Session() as session:
            # query: Query[entities.MemberResult] = session.query(entities.MemberResult).add_entity(entity)
            print(f"Before insert: {entity.member_result_id}")
            session.add(entity)
            session.commit()
            
            print(f"After insert: {entity.member_result_id}")

            return entity.member_result_id


    @classmethod
    def update_member_result(cls, member_result_id, old_prefix, new_prefix):
        with cls.Session() as session:
            object_to_update = session.query(entities.MemberResult).filter_by(member_result_id=member_result_id).first()

            if not object_to_update:
                return False
            
            old_url = str(object_to_update.house_url)
            print(f"Before update: {old_url}")
            object_to_update.house_url = old_url.replace(old_prefix, new_prefix)
            print(f"After update: {object_to_update.house_url}")

            old_url = str(object_to_update.tree_url)
            print(f"Before update: {old_url}")
            object_to_update.tree_url = old_url.replace(old_prefix, new_prefix)
            print(f"After update: {object_to_update.tree_url}")

            old_url = str(object_to_update.person_url)
            print(f"Before update: {old_url}")
            object_to_update.person_url = old_url.replace(old_prefix, new_prefix)
            print(f"After update: {object_to_update.person_url}")

            session.commit()

            return True



    @classmethod
    def get_member_results_ordered_by_test_date_desc(cls, member_id) -> Optional[List[entities.MemberResult]]:
        with cls.Session() as session:
            query: Query[entities.MemberResult] = session.query(entities.MemberResult) \
                .filter_by(member_id=member_id) \
                .order_by(desc(entities.MemberResult.test_time))

            result = query.all()

            return result


    #문장 결과는 마지막꺼만
    @classmethod
    def get_member_last_sentence_result(cls, member_id) -> Optional[str]:
        entity = None
        with cls.Session() as session:
            query: Query[entities.MemberResult] = session.query(entities.MemberResult) \
            .filter_by(member_id=member_id) \
            .order_by(desc(entities.MemberResult.test_time))
            entity = query.first()
        
        if not entity:
            return None
        
        with cls.Session() as session:
            query: Query[entities.Result] = session.query(entities.Result) \
            .filter_by(result_code=entity.result_code)

            result = query.first()

            if not result:
                return None
            else:
                return result.content