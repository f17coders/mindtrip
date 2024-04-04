from PIL import Image
from typing import Tuple

class Preprocessor:
    def __init__(self, path: str, target_size: Tuple[int, int]):
        self.path = path
        self.target_size = target_size


    def process(self):
        image = Image.open(fp=self.path, mode="r")
            
        # 투명 배경이 있는 이미지를 "RGB"로 변환하면서 흰색 배경으로 채우기
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            alpha = image.convert("RGBA").split()[-1]  # 알파 채널 추출
            
            # 새로운 "RGB" 이미지 생성 (흰색 배경)
            new_image = Image.new("RGB", image.size, "white")
            new_image.paste(image, mask=alpha)  # 투명 부분을 마스크로 사용하여 흰색 배경 이미지 위에 붙이기
            image = new_image

        # 이미지 크기 및 캔버스 크기 설정
        new_image = Image.new("RGB", self.target_size, "white")
        
        # 이미지 크기 조정
        # 이미지가 target_size보다 큰 경우, 이미지를 축소한다.
        if image.width > self.target_size[0] or image.height > self.target_size[1]:
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
        
        # 이미지가 target_size보다 작은 경우, 이미지의 중앙에 위치시키기 위해 offset 계산
        left_offset = (self.target_size[0] - image.width) // 2
        top_offset = (self.target_size[1] - image.height) // 2
        
        # 새 이미지 위에 기존 이미지 붙이기
        new_image.paste(image, (left_offset, top_offset))
        
        # 결과 이미지 저장 (여기서는 같은 경로에 덮어쓰기함)
        new_image.save(self.path)


# # 예제 사용법
# process_image("C:\\Users\\SSAFY\\Desktop\\Diagram.png")
