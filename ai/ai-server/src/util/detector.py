import os, subprocess

from util.preprocessor import Preprocessor

from typing import List, Dict, Union

class Detector():
    @classmethod
    def detect(
            cls,
            source: str,
            domain: str,
            project: str,
            name: str,
    ) -> List[Dict[str, Union[int, float]]]:
        # detection 결과는 PROJECT/NAME/에 저장된다.
        
        CONFIDENCE_THRESHOLD = 0.25
        INTERPRETER = "D:\\kirisame\\study\\extra\\ssafy\\02\\pjt02\\aiserver\\Scripts\\python.exe"
        # PROJECT = os.path.join("temp", f"{key}", "result")
        WEIGHT = os.path.join("weights", f"{domain}", "best.pt")

        try:
            subprocess.run(
                [
                    INTERPRETER, f"{os.path.join('app', 'detect.py')}",
                    "--weights", f"{WEIGHT}",  # pretrained-weight 파일 경로
                    "--img-size", f"{Preprocessor.TARGET_SIZE}",  # 이미지 크기
                    "--conf", f"{CONFIDENCE_THRESHOLD}",  # Confidence threshold
                    "--source", f"{source}",
                    "--project", f"{project}",
                    "--name", f"{name}",
                    "--save-txt",
                    "--save-conf",
                    "--exist-ok"
                ],
                # capture_output=True,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(e.stderr)
        except BaseException as e:
            import traceback
            traceback.print_exc()

        # 탐지된 게 없으면 labels/에 아무것도 저장되지 않는다.
        # 따라서 파일이 없는 경우 빈 리스트를 반환해야 한다.
        output_path = os.path.join(project, name, "labels", f"{domain}.txt")
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