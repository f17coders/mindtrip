from PIL import Image
import os

class Preprocessor:
    # def __init__(self, path: str, target_size: Tuple[int, int]):
    #     self.path = path
    #     self.target_size = target_size
    TARGET_SIZE = 1280

    @classmethod
    def process(cls, path):
        # 비동기 처리 특성 상 Redis에 캐싱된 후 임시파일을 바로 지우면
        # 여기서 없는 경로가 될 수 있다.
        # 이런 경우 그냥 return한다.
        if not os.path.exists(path):
            return

        # 이미지 열기
        img = Image.open(path)
        
        # 이미지가 투명한 배경을 가지고 있다면, 흰색 배경으로 변경
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, "WHITE")
            background.paste(img, img.split()[-1])
            img = background.convert("RGB")
        
        # 이미지의 원래 크기와 목표 크기
        original_width, original_height = img.size 
        
        # 이미지의 비율 유지하면서 크기 조정
        ratio = min(cls.TARGET_SIZE/original_width, cls.TARGET_SIZE/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 새 이미지에 흰색 배경 추가
        new_img = Image.new("RGB", (cls.TARGET_SIZE, cls.TARGET_SIZE), "WHITE")
        new_img.paste(img, ((cls.TARGET_SIZE - new_width) // 2, (cls.TARGET_SIZE - new_height) // 2))
        
        # 결과 이미지 저장
        new_img.save(path, quality=95)

