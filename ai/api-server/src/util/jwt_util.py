from fastapi import HTTPException
import jwt

from http import HTTPStatus
from dotenv import load_dotenv
import os

# for type hint
from typing import Dict, Union

# 디코딩할 JWT 토큰. 실제 사용 시에는 클라이언트로부터 받은 JWT 토큰을 사용합니다.
class JWTUtil:
    load_dotenv()
    __secret_key = os.getenv("JWT_SECRET_KEY")
    __hash_algorithm = os.getenv("HASH_ALGORITHM")
    __temp_token_lifetime = int(os.getenv("TEMP_TOKEN_LIFETIME_INT"))


    @classmethod
    def validate(cls, token: str) -> bool:
        return cls.__get_claims(token) != None


    @classmethod
    def __get_claims(cls, token: str) -> Dict[str, Union[str, int]]:
        try:
            token = token.removeprefix("Bearer ")
            claims = jwt.decode(
                jwt=token,
                key=cls.__secret_key,
                algorithms=[cls.__hash_algorithm]
            )

            # 디코딩된 JWT에서 claim 추출
            return claims
        except jwt.ExpiredSignatureError:
            raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 만료되었습니다.")
        except jwt.InvalidTokenError:
            raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")

    @classmethod
    def get_member_id(cls, token: str) -> str:
        claims = cls.__get_claims(token=token)
        if not claims:
            raise HTTPException(HTTPStatus.UNAUTHORIZED, "인증 정보가 유효하지 않습니다.")
        
        member_id = claims["memberId"]
        if not member_id:
            raise HTTPException(HTTPStatus.SERVICE_UNAVAILABLE, "토큰 정보를 읽어올 수 없습니다. 일시적인 오류일 수 있습니다.")
    
        return member_id


    @classmethod
    def create_temp_token(cls) -> str:
        import time
        from uuid import uuid4
        
        iat = int(time.time())
        exp = iat + cls.__temp_token_lifetime
        # iat는 기본적으로 현재 시간으로 추가된다.
        payload = {
            "memberId": str(uuid4()),
            'iat': iat,
            'exp': exp
        }

        return jwt.encode(payload=payload, key=cls.__secret_key, algorithm=cls.__hash_algorithm)