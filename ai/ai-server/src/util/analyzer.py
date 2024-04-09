from typing import List, Dict, Union
from type.domains import Domain

class Analyzer():
    @classmethod
    def analyze_detection(
            cls,
            domain: str,
            output: Dict[str, Union[int, float]]
    ) -> str:
        result =  None 
        if domain == Domain.HOUSE.value:
            result = cls.__analyze_detection_house(output)
        elif domain == Domain.TREE.value:
            result = cls.__analyze_detection_tree(output)
        elif domain == Domain.PERSON.value:
            result = cls.__analyze_detection_person(output)

        return result

    @classmethod
    def __analyze_detection_house(cls, output: List[Dict[str, Union[int, float]]]) -> str:
        detectable_classes = {
            "집전체": 0,
            "지붕": 1,
            "집벽": 2,
            "문": 3,
            "창문": 4,
            "굴뚝": 5,
            "연기": 6,
            "울타리": 7,
            "길": 8,
            "연못": 9,
            "산": 10,
            "나무": 11,
            "꽃": 12,
            "잔디": 13,
            "태양": 14
        }

        detected_classes = []
        for data in output:
            if "class" in data.keys():
                detected_classes.append(data["class"])

        ### 필수 항목의 존재 유무
        requirements = [1, 2, 3, 4]
        requirement_score = 0
        for detected_class in set(detected_classes):
            if detected_class in requirements:
                requirement_score += 1

        ### 존재 유무(각 1점)
        detail_score = 0

        # if 0 < 창문 <= 2 (너무 많으면 미친놈)
        detail_score += int(0 < detected_classes.count(detectable_classes["창문"]) <= 2)
        # if 0 < 문 <= 2 (너무 많으면 미친놈)
        detail_score += int(0 < detected_classes.count(detectable_classes["문"]) <= 2)
        # if 울타리 <= 2 (2개까지는 봐줌)
        detail_score += int(0 <= detected_classes.count(detectable_classes["울타리"]) <= 2)
        # if 굴뚝 <= 1 and 연기 <= 2
        detail_score += int(
            detected_classes.count(detectable_classes["굴뚝"]) <= 1 \
            and detected_classes.count(detectable_classes["연기"]) <= 2
        )
        # if 꽃 > 0 and 나무 > 0
        detail_score += int(
            detected_classes.count(detectable_classes["꽃"]) > 0 \
            and detected_classes.count(detectable_classes["나무"]) > 0
        )
        # if 꽃 < 5 (너무 많으면 X)
        detail_score += int(detected_classes.count(detectable_classes["꽃"]) <= 5)

        return requirement_score + detail_score

    @classmethod
    def __analyze_detection_tree(cls, output: Dict[str, Union[int, float]]) -> int:
        detectable_classes = {
            "나무전체": 0,
            "기둥": 1,
            "수관": 2,
            "가지": 3,
            "뿌리": 4,
            "나뭇잎": 5,
            "꽃": 6,
            "열매": 7,
            "그네": 8,
            "새": 9,
            "다람쥐": 10,
            "구름": 11,
            "달": 12,
            "별": 13,
        }

        detected_classes = []
        for data in output:
            if "class" in data.keys():
                detected_classes.append(data["class"])

        ### 필수 항목의 존재 유무(각 3점, 총 4점, )
        requirements = [1, 2, 4, 5]
        requirement_score = 0
        for detected_class in set(detected_classes):
            if detected_class in requirements:
                requirement_score += 1

        ### 세부사항 (각 1점)
        detail_score = 0
        detail_score += int(detected_classes.count(detectable_classes["열매"]) < 5)
        detail_score += int(detected_classes.count(detectable_classes["나뭇잎"]) <= 2)
        detail_score += int(detected_classes.count(detectable_classes["기둥"]) <= 2) # 새끼나무 = 애정 욕구

        return requirement_score + detail_score


    @classmethod
    def __analyze_detection_person(cls, output: Dict[str, Union[int, float]]) -> int:
        detectable_classes = {
            "사람전체": 0,
            "머리": 1,
            "얼굴": 2,
            "눈": 3,
            "코": 4,
            "입": 5,
            "귀": 6,
            "머리카락": 7,
            "목": 8,
            "상체": 9,
            "팔": 10,
            "손": 11,
            "다리": 12,
            "발": 13,
            "단추": 14,
            "주머니": 15,
            "운동화": 16,
        }

        detected_classes = []
        for data in output:
            if "class" in data.keys():
                detected_classes.append(data["class"])

        ### 필수 항목의 존재 유무
        requirements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        requirement_score = 0
        for detected_class in set(detected_classes):
            if detected_class in requirements:
                requirement_score += 1

        detail_score = 0
        detail_score += int(detected_classes.count(detectable_classes["단추"]) < 2)

        return requirement_score + detail_score
