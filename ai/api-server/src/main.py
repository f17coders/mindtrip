from fastapi import FastAPI, File, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from http import HTTPStatus

from util.redis_util import RedisUtil
from util.mariadb_util import MariaDBUtil
from util.s3_util import S3Util
from util.jwt_util import JWTUtil
from util.rabbitmq_util import RabbitMQUtil

import os
from pathlib import Path

import logging

# for type hint
from type.request import HtpAnswerRequest, HtpRegisterRequest
from fastapi import UploadFile
from type.entities import MemberResult
from type.domains import Domain


TOKEN_ALIAS = "Authorization"

import sys
sys.path.append("util")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["{{YOUR_ALLOW_ORIGINS}}"], # 특정 출처 허용
    allow_credentials={{YOUR_ALLOW_CREDENTIALS}}, # 인증 정보 요구 여부
    allow_methods=["{{YOUR_ALLOW_ORIGINS}}"],  # 특정 HTTP 메소드 허용
    allow_headers=["*"],  # 특정 HTTP 헤더 허용
    expose_headers=["*"]
)

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 로거 생성
logger = logging.getLogger(__name__)


# Health check용
@app.get("/htp/hello")
def hello():
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": "hello"
        }
    )

@app.get("/htp/v0/temp_token")
def temp_token():
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            TOKEN_ALIAS: JWTUtil.create_temp_token()
        }
    )

# v0는 토큰 없음, v1은 토큰 있음(임시 토큰 포함)
@app.post("/htp/v1/test/{domain}")
async def htp_test(
    domain: str,
    token=Header(..., alias=TOKEN_ALIAS),
    file: UploadFile=File( ..., alias="file"),
):  
    logger.info(f"token: {token}")
    if not token:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="인증 정보가 없습니다."
        )

    member_id = JWTUtil.get_member_id(token)
    logger.info(f"member_id: {member_id}")
    if not member_id:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="인증 정보가 유효하지 않습니다."
        )

    if domain not in Domain.get_value_list():
        logger.info(f"도메인 없음: {domain}")
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="그런 검사 항목은 없습니다."
        )

    if not file:
        logger.info(f"파일 없음")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="파일이 없습니다."
        )

    if not __check_format(file):
        logger.error(f"파일명: {file.filename}, 확장자명: {Path(file.filename).suffix}")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"파일명: {file.filename}, 확장자명: {Path(file.filename).suffix} 그림 파일은 .png만 업로드할 수 있습니다."
        )
    logger.info(f"파일명: {file.filename}, 확장자명: {Path(file.filename).suffix}")

    if not member_id:
        logger.info(f"토큰에 memberId 없음")
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"사용자 정보가 없습니다."
        )

    # 1. 집 그림을 UploadFile(input)로 받고 임시 파일로 저장한다.
    temp_file_path = __get_temp_file_path(
        domain=domain,
        file=file,
        key=member_id
    )
    if os.path.exists(temp_file_path):
        raise HTTPException(HTTPStatus.ACCEPTED, "이미 처리 중입니다.")
    logger.info(f"temp_file_path: {temp_file_path}")

    contents = await file.read()
    with open(temp_file_path, "wb") as buffer:
        buffer.write(contents)
        buffer.flush()
    logger.info("파일 업로드 끝")

    # 2. S3에 저장할 원본 이미지를 따로 저장한다.
    origin_path = __get_original_file_path(
        domain=domain,
        file=file,
        key=member_id
    )
    with open(origin_path, "wb") as buffer:
        buffer.write(contents)
        buffer.flush()
    del contents

    logger.info(f"temp_file_path: {temp_file_path}")
    RabbitMQUtil.publish(
        member_id=member_id,
        domain=domain,
        temp_file_path=os.path.abspath(temp_file_path),
        project=os.path.abspath(os.path.join(TEMP_DIR, member_id, "result")),
        name = domain,
    )

    return JSONResponse(
        content={
            "message": f"{domain} 이미지 파일이 업로드되었습니다."
        },
        status_code=HTTPStatus.OK
    )


@app.get("/htp/v0/question/all")
def question():
    domain_list = Domain.get_value_list()
    
    all_questions = {}
    
    for domain in domain_list:
        question_list = MariaDBUtil.get_questions(domain)
        questions = []
        for question in question_list:
            obj = {}
            obj["question_id"] = question.question_id
            obj["content"] = question.content

            choice_list = MariaDBUtil.get_choices(question_id=question.question_id)
            choices = []
            for choice in choice_list:
                choices.append(
                    {
                        "choice_id": choice.choice_id,
                        "content" : choice.content
                    }
                )
            obj["choices"] = choices  
            questions.append(obj)

        all_questions[domain] = questions

    return all_questions


@app.get("/htp/v0/question/{domain}")
def question(domain: str):
    question_list = MariaDBUtil.get_questions(domain)
    questions = []
    for question in question_list:
        obj = {}
        obj["question_id"] = question.question_id
        obj["content"] = question.content

        choice_list = MariaDBUtil.get_choices(question_id=question.question_id)
        choices = []
        for choice in choice_list:
            choices.append(
                {
                    "choice_id": choice.choice_id,
                    "content" : choice.content
                }
            )
        obj["choices"] = choices
        
        questions.append(obj)

    return questions


@app.post("/htp/v1/answer")
def answer(
    req: HtpAnswerRequest,
    token: str=Header(..., alias=TOKEN_ALIAS)
):
    if not token:
        logger.info(f"토큰 없음")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 없습니다.")

    member_id = JWTUtil.get_member_id(token)
    if not member_id:
        logger.info(f"토큰에 memberId 없음")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    logger.info(f"/htp/v1/answer token: {token}")
    logger.info(f"member in answer(): {member_id}")
    logger.info(f"is_registered: {token.startswith('Bearer')}")

    # 요청으로 들어온 답변 데이터로부터
    prev_scores = RedisUtil.get(member_id)
    
    if not prev_scores:
        import time

        key = f"count{member_id}"
        RedisUtil.put(key, 0)
        # count = 0
        while (prev_scores := RedisUtil.get(member_id)) == None or len(prev_scores) != 3:
            if RedisUtil.get(key) >= 300:
            # if count >= 300:
                break

            RedisUtil.put(key, RedisUtil.get(key) + 1)
            # count += 1
            time.sleep(1)
        RedisUtil.delete(key)
        del key

        if prev_scores := RedisUtil.get(member_id) == None or len(prev_scores) != 3:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "먼저 HTP 검사를 실시해주세요.")

    logger.info(f"prev_scores in answer(): {prev_scores}")

    answers = req.answer
    # 제출은 됐는데 검출 후 결과 캐싱이 끝나기 전 여기에 도달하면 문제가 생긴다.
    prev_scores = RedisUtil.get(member_id)
    for key in answers.keys():
        if key not in prev_scores.keys():
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, f"{key} 검사가 완료되지 않았습니다.")

    # 질문 점수를 이미지 분석 점수에 더한다.
    for domain in answers.keys():
        for answer in answers[domain]:
            score = MariaDBUtil.get_score(choice_id=answer["choice_id"])
            prev_scores[domain] = prev_scores[domain] + score

    # { "house": 3, "tree": 4, "person": 5 }
    # 앞에서 구해진 총 점수로부터 결과 코드를 얻는다.
    thresholds = {
        Domain.HOUSE: round(4/2 + 6/2 + 1.5*3, 0),
        Domain.TREE: round(4/2 + 3/2 + 1.5*5, 0),
        Domain.PERSON: round(13/2 + 1/2 + 1.5*6, 0),
    }
    code = "".join(
        [
            str(int(prev_scores[Domain.HOUSE.value] >= thresholds[Domain.HOUSE])),
            str(int(prev_scores[Domain.TREE.value] >= thresholds[Domain.TREE])),
            str(int(prev_scores[Domain.PERSON.value] >= thresholds[Domain.PERSON]))
        ]
    )

    # S3에 이미지를 저장한 후 저장된 이미지의 url을 얻는다.
    urls = {}
    for domain in Domain.get_value_list():
        logger.info(f"URL_DOMAIN: {domain}")

        src_dir = __get_original_dir_path(domain, key=member_id)
        if not os.path.exists(src_dir) or not os.listdir(src_dir):
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "파일이 업로드되지 않았습니다. 검사를 다시 수행해주세요.")

        extension = None
        for file_name in os.listdir(src_dir):
            if file_name.startswith(f"{domain}_original"):
                extension = Path(file_name).suffix
                break

        if not extension:
            logger.info(f"유효하지 않은 파일: {os.listdir(src_dir)}")
            raise HTTPException(HTTPStatus.BAD_REQUEST, "검사를 다시 수행해주세요.")

        src_path = os.path.join(src_dir, f"{domain}_original{extension}")
        dest_path = "/".join([member_id, f"{domain}{extension}"])
        logger.info(f"from {src_path} to {dest_path}")
        url = S3Util.upload_to_s3(
            src=src_path,
            dest=dest_path
        )
        urls[f"{domain}_url"] = url

    # 기존 회원이면
    logger.info(f"기존 회원인가?: {token.startswith('Bearer')}")
    if token.startswith("Bearer"):
        logger.info(f"member_id: {member_id}")

        if not MariaDBUtil.exists_member_by_member_id(member_id=int(member_id)):
            raise HTTPException(HTTPStatus.BAD_REQUEST, "존재하지 않는 사용자 정보입니다.")

        logger.info(f"urls: {urls}")
        entity = MemberResult(
            result_code = code,
            member_id = member_id,
            house_url = urls[f"house_url"],
            tree_url = urls[f"tree_url"],
            person_url = urls[f"person_url"]
        )
        logger.info("GET ENTITY")

        # 새로운 결과 데이터를 생성해 저장한 후 저장된 데이터의 id 값을 얻는다.
        logger.info(f"New member result {member_id}: {entity}")
        result_id = MariaDBUtil.insert_member_result(entity)

        # S3에서 앞에서 얻은 결과 id에 맞는 경로로 옮긴다.
        S3Util.move(old_prefix=f"{member_id}", new_prefix=f"{member_id}/{result_id}")
        logger.info("S3 moving")

        # 옮겨진 경로로 사용자 결과 데이터를 갱신한다.
        result = MariaDBUtil.update_member_result(result_id, old_prefix=f"{member_id}", new_prefix=f"{member_id}/{result_id}")
        logger.info(f"update member_result({result_id}, {member_id}, {member_id}/{result_id})")
        logger.info(f"{'SUCC' if result else 'FAIL'}")

        # 갱신된 결과 데이터에 맞게 사용자의 마을을 바꿔준다.
        new_village_id = MariaDBUtil.get_village_id_by_result_code(code)
        logger.info(f"Before update member: {MariaDBUtil.get_member_by_member_id(int(member_id))}")
        MariaDBUtil.update_member(member_id=member_id, new_village_id=new_village_id)
        logger.info(f"After update member: {MariaDBUtil.get_member_by_member_id(int(member_id))}")

        # 임시파일과 레디스 정리
        import shutil
        target = os.path.join(TEMP_DIR, member_id)
        if os.path.exists(target):
            print(f"{target} exists.")
            shutil.rmtree(target)
        
        RedisUtil.delete(member_id)
    else:
        # Redis에 검사 결과 캐싱
        data = {
            "code": code,
            "urls": urls
        }
        logger.info(f"레디스에 넣는다아아아아아 {member_id}:{data}")
        RedisUtil.put(member_id, data)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": code
        }
    )


@app.get("/htp/v1/result/sentence")
def result_sentence(
    token=Header(..., alias=TOKEN_ALIAS)
):
    if not token:
        logger.info("No token")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 없습니다.")

    member_id = JWTUtil.get_member_id(token)
    logger.debug(f"member in result_sentence(): {member_id}")
    if not member_id:
        logger.info("Invalid authentication")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    if token.startswith("Bearer"):
        logger.info("기존 회원이라면 여기로")
        village_id = MariaDBUtil.get_village_id_by_member_id(member_id=int(member_id))
        sentence = MariaDBUtil.get_result_sentence_by_village_id(village_id=village_id)

        if sentence == None:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "먼저 HTP 검사를 진행해주세요.")

        return {"data": sentence}

    logger.info("신규 회원이라면 여기로")
    data = None
    count = 0
    import time
    while (data := RedisUtil.get(member_id)) == None:
        if count >= 20:
            break
        count += 1
        time.sleep(1)
    
    logger.info(f"data in /htp/v1/result/sentence {data}")
    if not data:
        logger.info("/htp/v1/result/sentence: No data")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사 데이터가 없습니다.")

    if "code" not in data.keys():
        count = 0
        import time
        while "code" not in (data := RedisUtil.get(member_id)).keys():
            print(f"data in sentence: {data}")
            if count >= 20:
                break
            count += 1
            time.sleep(1)
        logger.info("/htp/v1/result/sentence: No code")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "먼저 HTP 검사를 실시해주세요.")

    code = data["code"]
    result = MariaDBUtil.get_result_sentence(code)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": result
        }
    )

@app.get("/htp/v1/result/village") 
async def result_village(
    token=Header(..., alias=TOKEN_ALIAS),
):
    if not token:
        logger.info("No token")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 없습니다.")

    member_id = JWTUtil.get_member_id(token)
    logger.debug(f"member in result_village(): {member_id}")
    if not member_id:
        logger.info("Invalid authentication")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    if token.startswith("Bearer"):
        village_id = MariaDBUtil.get_village_id_by_member_id(member_id=int(member_id))
        village = MariaDBUtil.get_village_by_village_id(village_id)
        if village == None:
            raise HTTPException(HTTPStatus.UNPROCESSABLE_ENTITY, "먼저 HTP 검사를 진행해주세요.")

        return {"data": village}



    logger.info("신규 회원이라면 여기로")
    data = None
    count = 0
    import time
    while (data := RedisUtil.get(member_id)) == None:
        if count >= 20:
            break
        count += 1
        time.sleep(1)
    logger.info(f"data in /htp/v1/result/village: {data}")
    if not data:
        logger.info(f"No data")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사 데이터가 없습니다.")

    count = 0
    import time
    while "code" not in (data := RedisUtil.get(member_id)).keys():
        if count >= 20:
            break
        count += 1
        time.sleep(1)

    if "code" not in data.keys():
        logger.info(f"/htp/v1/result/village: No code")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "먼저 HTP 검사를 실시해주세요.")

    code = data["code"]
    logger.info(f"code in result_village: {code}")
    result = MariaDBUtil.get_result_village(code)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content = {
            "data": result
        }
    )


@app.post("/htp/v1/register")
def register(
    req: HtpRegisterRequest,
    token=Header(..., alias=TOKEN_ALIAS)
):
    if not token:
        logger.info("No token")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 없습니다.")

    if not req or not req.member_id:
        logger.info("No memberId")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "잘못된 요청입니다.")

    temp_member_id = JWTUtil.get_member_id(token) # UUID로 구성되는 임시 member_id
    member_id = req.member_id # 실제 회원가입된 사용자의 member_id
    
    logger.info(f"temp_member_id: {temp_member_id}")
    if not temp_member_id:
        logger.info("No memberId in token")
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    data = RedisUtil.get(temp_member_id)
    if not data:
        logger.info(f"No key in Redis: {temp_member_id}")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "먼저 HTP 검사를 실시해주세요.")
    RedisUtil.delete(temp_member_id)

    if "code" not in data.keys():
        logger.info(f"There is no result code: {data}")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사 데이터가 없거나 손상되었습니다.")
    result_code = data["code"]
    if not result_code:
        logger.info(f"No result code")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사 결과 코드가 없습니다.")

    urls = data["urls"]
    if not urls:
        logger.info("No URL")
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사를 위해 제출된 이미지가 없습니다. 다시 검사를 실시해주세요.")

    logger.info(f"Given urls: {urls}")

    entity = MemberResult(
        result_code = result_code,
        member_id = member_id,
        house_url = urls[f"house_url"],
        tree_url = urls[f"tree_url"],
        person_url = urls[f"person_url"]
    )
    new_village_id = MariaDBUtil.get_village_id_by_result_code(result_code)
    result_id = MariaDBUtil.insert_member_result(entity)
    S3Util.move(old_prefix=f"{temp_member_id}/", new_prefix=f"{member_id}/{result_id}/")
    result = MariaDBUtil.update_member_result(result_id, old_prefix=f"{temp_member_id}", new_prefix=f"{member_id}/{result_id}")
    MariaDBUtil.update_member(member_id=member_id, new_village_id=new_village_id)

    # 임시파일과 레디스 정리
    import shutil
    target = os.path.join(TEMP_DIR, temp_member_id)
    if os.path.exists(target):
        print(f"{target} exists.")
        shutil.rmtree(target)
    RedisUtil.delete(temp_member_id)

    if result:
        return JSONResponse(
            content={
                "message": "검사 결과 등록 성공"
            },
            status_code=HTTPStatus.CREATED
        )
    else:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "검사 결과 등록 중 오류가 발생했습니다.")


@app.get("/htp/v1/member/htp-results")
def htp_member_results(
    token=Header(..., alias=TOKEN_ALIAS)
):
    member_id = JWTUtil.get_member_id(token)
    ordered_results = MariaDBUtil.get_member_results_ordered_by_test_date_desc(member_id)

    return ordered_results


@app.get("/htp/v1/member/htp-last-result")
def htp_last_member_result(
    token=Header(..., alias=TOKEN_ALIAS)
):
    member_id = JWTUtil.get_member_id(token)
    last_result = MariaDBUtil.get_last_member_sentence_result(member_id)

    return last_result


def __get_temp_file_path(
        domain: str,
        file: UploadFile,
        key: str
) -> str:
    if not domain:
        raise HTTPException(400, "도메인 정보가 없습니다.")

    if not file:
        raise HTTPException(400, "파일 정보가 없습니다.")

    dir_path = os.path.join(TEMP_DIR, key, "data")
    if not os.path.exists(path=dir_path):
        os.makedirs(name=dir_path, mode=555)

    return os.path.join(
        dir_path,
        f"{domain}{Path(file.filename).suffix}"
    )


def __get_original_file_path(
        domain: str,
        file: UploadFile,
        key: str
) -> str:
    if not domain:
        raise HTTPException(400, "도메인 정보가 없습니다.")

    if not file:
        raise HTTPException(400, "파일 정보가 없습니다.")

    dir_path = os.path.join(TEMP_DIR, key, "data")
    if not os.path.exists(path=dir_path):
        os.makedirs(name=dir_path, mode=555)

    return os.path.join(
        dir_path,
        f"{domain}_original{Path(file.filename).suffix}"
    )


def __get_original_dir_path(
        domain: Domain,
        key: str
) -> str:
    if not domain:
        return None

    dir_path = os.path.join(TEMP_DIR, key, "data")
    if not os.path.exists(path=dir_path):
        os.makedirs(name=dir_path, mode=555)

    return dir_path


def __check_format(file: UploadFile) -> bool:
    extension = Path(file.filename).suffix.lower()
    extensions = set([
        ".jpg",
        ".jpeg",
        ".png"
    ])

    return extension in extensions


def __run():
    global TEMP_DIR
    TEMP_DIR = "temp"

    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=54354)


if __name__ == "__main__":
    __run()
