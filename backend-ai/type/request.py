from pydantic import BaseModel

from fastapi import UploadFile, File
from typing import List, Dict


class HtpAnswerRequest(BaseModel):
    answer: Dict[str, List[Dict[str, int]]]


class HtpRegisterRequest(BaseModel):
    member_id: str
