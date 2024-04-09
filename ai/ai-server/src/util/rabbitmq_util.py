import pika
import os, dotenv
import json 
from util.detector import Detector
from util.preprocessor import Preprocessor
from util.redis_util import RedisUtil


class RabbitMQUtil():
    dotenv.load_dotenv()
    __queue = os.getenv("RABBITMQ_QUEUE_NAME")
    __host = os.getenv("RABBITMQ_HOST")
    __username = os.getenv("RABBITMQ_DEFAULT_USER")
    __password = os.getenv("RABBITMQ_DEFAULT_PASS")
    __port = os.getenv("RABBITMQ_PORT_AMQP")

    @classmethod
    def __get_parameters(cls):
        return pika.ConnectionParameters(
            host=cls.__host,
            credentials=pika.PlainCredentials(
                username=cls.__username,
                password=cls.__password,
            ),
            port=cls.__port
        )

    @classmethod
    def consume(cls):
        with pika.BlockingConnection(cls.__get_parameters()) as connection:
            channel = connection.channel()

            # 큐 생성 (이미 존재하면 무시)
            channel.queue_declare(queue=cls.__queue)

            # 큐로부터 메시지를 받기 위해 구독
            channel.basic_consume(
                queue=cls.__queue,
                on_message_callback=cls.__callback,
                auto_ack=True
            )

            print(' [*] Waiting for messages. To exit press CTRL+C')
            channel.start_consuming()


    @staticmethod
    def __callback(ch, method, properties, body):
        data = json.loads(body)
        print(f" [x] Received {data}")
        member_id = data["member_id"]
        temp_file_path = data["temp_file_path"]
        domain = data["domain"]
        project = data["project"]
        name = data["name"]

        # Detector.detect(source=source, domain=domain, key=key)
        Preprocessor.process(path=temp_file_path)

        # 4. 전처리된 이미지로 detection을 수행하고 dectection 결과 데이터 얻기
        # detection 결과는 json 객체로 받는다.
        print(f"call Detector.detect({temp_file_path}, {domain}, {project}, {name})")
        output = Detector.detect(
            source=temp_file_path,
            domain=domain,
            project=project,
            name=name
        )
        print(f"output: {output}")

        # 6. 검출 데이터 결과 얻기
        from util.analyzer import Analyzer
        detection_result: int = Analyzer.analyze_detection(domain=domain, output=output)
        
        # 7. 검출 결과 캐싱
        prev_scores = RedisUtil.get(member_id)
        print(f"prev: {prev_scores}")
        if prev_scores:
            prev_scores[domain] = detection_result
            RedisUtil.put(key=member_id, value=prev_scores)
            print("prev_scores 있다!")
        else:
            RedisUtil.put(
                key=member_id,
                value={
                    domain: detection_result
                }
            )
        # { "house": 3, "tree": 4, "person": 5 }
        print(f"{domain} cached: {RedisUtil.get(member_id)}")

