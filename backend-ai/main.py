from fastapi import FastAPI, File, HTTPException, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from http import HTTPStatus

from util.redis_util import RedisUtil
from util.mariadb_util import MariaDBUtil
from util.s3_util import S3Util
from util.jwt_util import JWTUtil
from util.preprocess import Preprocessor

import subprocess
import os
from pathlib import Path

import logging

# for type hint
from type.request import HtpAnswerRequest, HtpRegisterRequest
from typing import Dict, Union, List
from fastapi import UploadFile
from type.entities import Question, MemberResult
from type.domains import Domain, DBDomain



TOKEN_ALIAS = "Authorization"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["http://localhost:5173", "https://mindtrip.site"],  # 특정 도메인 허용
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],  # 특정 HTTP 메소드 허용
    allow_headers=["*"],  # 특정 HTTP 헤더 허용
    expose_headers=["*"]
)

# 로거 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 로거 생성
logger = logging.getLogger(__name__)



@app.get("/api/htp/hello")
def hello_api():
    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": "hello-api" 
        }
    )

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


# v0는 토큰 없음, v1은 임시 토큰, v2는 진짜 토큰
@app.post("/htp/v1/test/{domain}")
async def htp_test(
    domain: str,
    token=Header(..., alias=TOKEN_ALIAS),
    file: UploadFile=File( ..., alias="file"),
):  
    logging.info(f"token: {token}")
    temp_member_id = JWTUtil.get_member_id(token)
    res = await __htp_test(
        domain=domain,
        member_id=temp_member_id,
        file=file
    )

    return JSONResponse(
        status_code=res["status_code"],
        content=res["content"]
    )

async def __htp_test(domain: str, member_id, file: UploadFile):
    if domain not in Domain.get_value_list():
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail="그런 검사 항목은 없습니다."
        )
    
    if not file:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="파일이 없습니다."
        )

    if not __check_format(file):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"파일명: {file.filename}, 확장자명: {Path(file.filename).suffix} 그림 파일은 .png만 업로드할 수 있습니다."
        )

    if not member_id:
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
    logger.debug(f"temp_file_path: {temp_file_path}")

    contents = await file.read()
    with open(temp_file_path, "wb") as buffer:
        buffer.write(contents)
        buffer.flush()

    # 2. 이미지를 S3에 저장하기 위해 원본 이미지를 별도로 저장한다.
    # S3 상의 경로는 결과 값의 id로 정해지는데, 아직 결과가 안 나와서 어쩔 수 없음...
    origin_path = __get_original_file_path(
        domain=domain,
        file=file,
        key=member_id
    )
    with open(origin_path, "wb") as buffer:
        buffer.write(contents)
        buffer.flush()
    del contents
    
    # 3. 받은 집 그림 이미지 전처리를 수행한다.
    __preprocess(path=temp_file_path)

    # 4. 전처리된 이미지로 detection을 수행하고 dectection 결과 데이터 얻기
    # detection 결과는 json 객체로 받는다.
    logging.info(f"call __detect({temp_file_path}, {domain}, {member_id})")
    output = __detect(
        source=temp_file_path,
        domain=domain,
        key=member_id
    )
    logging.info(f"output: {output}")

    # 6. 검출 데이터 결과 얻기
    detection_result: str = __analyze_detection(domain=domain,output=output)
    
    # 7. 검출 결과 캐싱
    prev_scores = RedisUtil.get(member_id)
    logging.info(f"prev: {prev_scores}")
    if prev_scores:
        prev_scores[domain] = detection_result
        RedisUtil.put(key=member_id, value=prev_scores)
        logging.info("prev_scores 있다!")
    else:
        RedisUtil.put(
            key=member_id,
            value={
            domain: detection_result
        })

    logging.info(RedisUtil.get(member_id))

    return {
        "status_code": HTTPStatus.OK,
        "content": {
            "message": f"{domain} test 완료"
        }
    }


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


#@app.post("/htp/v1/answer")
def answer(
    req: HtpAnswerRequest,
    temp_token=Header(..., alias=TOKEN_ALIAS)
):
    answers = req.answer
    key = JWTUtil.get_member_id(temp_token)
    prev_scores = RedisUtil.get(key)
    logging.info(f"prev_scores in answer(): {prev_scores}")
    
    for domain in answers.keys():
        for answer in answers[domain]:
            score = MariaDBUtil.get_score(choice_id=answer["choice_id"])
            prev_scores[domain] = int(prev_scores[domain]) + score

    thresholds = {
        Domain.HOUSE: 6,
        Domain.TREE: 12,
        Domain.PERSON: 12,
    }

    code = "".join(
        [
            str(int(prev_scores[Domain.HOUSE.value] >= thresholds[Domain.HOUSE])),
            str(int(prev_scores[Domain.TREE.value] >= thresholds[Domain.TREE])),
            str(int(prev_scores[Domain.PERSON.value] >= thresholds[Domain.PERSON]))
        ]
    )


    # S3에 이미지를 저장한다.
    # s3://a303-bucket/temp_member_id/
    urls = {}
    for domain in Domain.get_value_list():
        url = S3Util.upload_to_s3(
            src=os.path.join(__get_original_dir_path(domain, key=key), f"{domain}_original.png",),
            dest="/".join([key, f"{domain}.png"])
        )
        # urls.append({f"{domain}_url": url})
        urls[f"{domain}_url"] = url
        
    logging.info(f"urls: {urls}")

    data = {
        "code": code,
        "urls": urls
    }
    RedisUtil.put(key, data)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": code
        }
    )

@app.post("/htp/v1/answer")
def answer(
    req: HtpAnswerRequest,
    token=Header(..., alias=TOKEN_ALIAS)
):
    logging.info(f"/htp/v1/answer token: {token}")
    answers = req.answer
    key = JWTUtil.get_member_id(token)
    logging.info(f"key in answer(): {key}")
    prev_scores = RedisUtil.get(key)
    logging.info(f"prev_scores in answer(): {prev_scores}")
    
    for domain in answers.keys():
        for answer in answers[domain]:
            score = MariaDBUtil.get_score(choice_id=answer["choice_id"])
            prev_scores[domain] = int(prev_scores[domain]) + score

    thresholds = {
        Domain.HOUSE: 6,
        Domain.TREE: 12,
        Domain.PERSON: 12,
    }

    code = "".join(
        [
            str(int(prev_scores[Domain.HOUSE.value] >= thresholds[Domain.HOUSE])),
            str(int(prev_scores[Domain.TREE.value] >= thresholds[Domain.TREE])),
            str(int(prev_scores[Domain.PERSON.value] >= thresholds[Domain.PERSON]))
        ]
    )


    # S3에 이미지를 저장한다.
    # s3://a303-bucket/temp_member_id/
    urls = {}
    for domain in Domain.get_value_list():
        url = S3Util.upload_to_s3(
            src=os.path.join(__get_original_dir_path(domain, key=key), f"{domain}_original.png",),
            dest="/".join([key, f"{domain}.png"])
        )
        # urls.append({f"{domain}_url": url})
        urls[f"{domain}_url"] = url
        
    logging.info(f"urls: {urls}")

    data = {
        "code": code,
        "urls": urls
    }
    RedisUtil.put(key, data)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": code
        }
    )


@app.get("/htp/v1/result/sentence")
def result_sentence(
    temp_token=Header(..., alias=TOKEN_ALIAS)
):
    key = JWTUtil.get_member_id(temp_token)
    code = RedisUtil.get(key)["code"]
    result = MariaDBUtil.get_result_sentence(code)

    return JSONResponse(
        status_code=HTTPStatus.OK,
        content={
            "data": result
        }
    )


@app.get("/htp/v1/result/village")
def result_village(
    temp_token=Header(..., alias=TOKEN_ALIAS),
):
    key = JWTUtil.get_member_id(temp_token)
    code = RedisUtil.get(key)["code"]
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
    temp_member_id = JWTUtil.get_member_id(token) #UUID
    member_id = req.member_id # 실제 회원가입된 사용자 ID
    
    if not temp_member_id:
        raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    data = RedisUtil.get(temp_member_id)
    if not data:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "먼저 HTP 검사를 실시해주세요.")
    
    result_code = data["code"]
    if not result_code:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사 결과 코드가 없습니다.")

    # {'house_url': '1234', 'tree_url': '1234', 'person_url': '1234'}
    urls = data["urls"]
    if not urls:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "HTP 검사를 위해 제출된 이미지가 없습니다. 다시 검사를 실시해주세요.")

    logging.info(f"Given urls: {urls}")

    # 여기서 DB에 넣자
    entity = MemberResult(
        result_code = result_code,
        member_id = member_id,
        house_url = urls[f"house_url"],
        tree_url = urls[f"tree_url"],
        person_url = urls[f"person_url"]
    )
    result_id = MariaDBUtil.insert_member_result(entity)

    S3Util.move(old_prefix=f"{temp_member_id}", new_prefix=f"{member_id}/{result_id}")
    result = MariaDBUtil.update_member_result(result_id, old_prefix=f"{temp_member_id}", new_prefix=f"{member_id}/{result_id}")

    if result:
        return JSONResponse(
            content={
                "message": "검사 결과 등록 성공"
            },
            status_code=HTTPStatus.CREATED
        )
    else:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, "검사 결과 등록 중 오류가 발생했습니다.")


def __preprocess(path: str):
    global IMAGE_SIZE
    IMAGE_SIZE = 1280

    processor = Preprocessor(path=path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    processor.process()


def __detect(
        source: str,
        domain: str,
        key: str
) -> List[Dict[str, Union[int, float]]]:
    # detection 결과는 PROJECT/NAME/에 저장된다.
    INTERPRETER = "python"
    
    CONFIDENCE_THRESHOLD = 0.25
    INTERPRETER = "python"
    logging.info(f'source: {source}, Detect {domain}, key: {key}')
    PROJECT = os.path.join("temp", f"{key}", "result")
    NAME = f"{domain}"
    WEIGHT = os.path.join("weights", f"{domain}", "best.pt")

    result = None

    # if not os.path.exists(INTERPRETER):
    #     logging.info(f"Check interpreter path: {INTERPRETER}")
    #     raise HTTPException(503, "죄송합니다. 지금은 AI 검사를 이용할 수 없습니다. 관리자에게 문의해주세요.")

    if not os.path.exists(WEIGHT):
        logging.info(f"Check weights path: {WEIGHT}")
        raise HTTPException(503, "지금은 AI 검사를 이용할 수 없습니다. 관리자에게 문의해주세요.")

    if not os.path.exists(source):
        logging.info(f"Check data path: {source}")
        raise HTTPException(503, "지금은 AI 검사를 이용할 수 없습니다. 관리자에게 문의해주세요.")
    
    #INTERPRETER = "D:\\kirisame\\study\\extra\\ssafy\\02\\pjt02\\aisrv\\Scripts\\python.exe"
    try:
        result = subprocess.run(
            [
                INTERPRETER, f"{os.path.join('app', 'detect.py')}",
                "--weights", f"{WEIGHT}",  # pretrained-weight 파일 경로
                "--img-size", f"{IMAGE_SIZE}",  # 이미지 크기
                "--conf", f"{CONFIDENCE_THRESHOLD}",  # Confidence threshold
                "--source", f"{source}",
                "--project", f"{PROJECT}",
                "--name", f"{NAME}",
                "--save-txt",
                "--save-conf",
                "--exist-ok"
            ],
            # capture_output=True,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )

        logging.debug(result.stderr)

    except subprocess.CalledProcessError as e:
        print(e.stderr)
        logging.error(e)
    except BaseException as e:
        raise HTTPException(500, "AI 검사에 실패했습니다.")

    if not result:
        raise HTTPException(500, "알 수 없는 오류로 AI 검사에 실패했습니다.")

    if result.returncode != 0:
        raise HTTPException(500, "검사 결과 분석 실패")


    # 탐지된 게 없으면 labels/에 아무것도 저장되지 않는다.
    # 따라서 파일이 없는 경우 빈 리스트를 반환해야 한다.
    output_path = os.path.join(PROJECT, NAME, "labels", f"{domain}.txt")
    if not os.path.exists(output_path):
        return []

    json_output = []
    with open(file=output_path, mode="rt") as f:
        for line in f.readlines():
            class_id, x_center, y_center, width, height, confidence = line.split()
            class_id = int(class_id)
            width, height = float(width), float(height)
            confidence = float(confidence)

            # 검출된 객체의 좌상단 위치
            x, y = float(x_center) - (width/2), float(y_center) - (height/2)

            json_object = {
                "class": int(class_id),
                "x": x,
                "y": y,
                "width": float(width),
                "height": float(height),
                "confidence": float(confidence)
            }

            json_output.append(json_object)

    return json_output


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
    last_result = MariaDBUtil.get_member_last_sentence_result(member_id)

    return last_result


def __analyze_detection(
        domain: str,
        output: Dict[str, Union[int, float]]
) -> str:
    result =  None 
    if domain == Domain.HOUSE.value:
        result = __analyze_detection_house(output)
    elif domain == Domain.TREE.value:
        result = __analyze_detection_tree(output)
    elif domain == Domain.PERSON.value:
        result = __analyze_detection_person(output)

    logging.info(f"{domain} 검출 결과: {result}")

    return result


def __analyze_detection_house(output: Dict[str, Union[int, float]]) -> str:
    classes = {
        0: "집전체",
        1: "지붕",
        2: "집벽",
        3: "문",
        4: "창문",
        5: "굴뚝",
        6: "연기",
        7: "울타리",
        8: "길",
        9: "연못",
        10: "산",
        11: "나무",
        12: "꽃",
        13: "잔디",
        14: "태양"
    }

    score = 0
    score_threshold = 3

    return "1" if score >= score_threshold else "0"


def __analyze_detection_tree(json_output: Dict[str, Union[int, float]]) -> str:
    classes = [
        "나무전체",
        "기둥",
        "수관",
        "가지",
        "뿌리",
        "나뭇잎",
        "꽃",
        "열매",
        "그네",
        "새",
        "다람쥐",
        "구름",
        "달",
        "별"
    ]

    score = 0
    score_threshold = 3
    # 일정 점수 미만이면 GOOD, 아니면 BAD

    return "1" if score >= score_threshold else "0"


def __analyze_detection_person(json_output: Dict[str, Union[int, float]]) -> str:
    classes = [
        "사람전체",
        "머리",
        "얼굴",
        "눈",
        "코",
        "입",
        "귀",
        "머리카락",
        "목",
        "상체",
        "팔",
        "손",
        "다리",
        "발",
        "단추",
        "주머니",
        "운동화",
    ]
    score = 0
    score_threshold = 3
    # 일정 점수 미만이면 GOOD, 아니면 BAD

    return "1" if score >= score_threshold else "0"


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
    return Path(file.filename).suffix == ".png" or Path(file.filename).suffix == ".PNG"


def __run():
    global TEMP_DIR
    TEMP_DIR = "temp"

    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=54353)


if __name__ == "__main__":
    # RedisUtil.clear()
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    print(os.getenv("REDIS_PORT_INT"))

    __run()


