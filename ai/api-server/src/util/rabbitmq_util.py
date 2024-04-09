import pika
import os, dotenv
import json 



class RabbitMQUtil():
    dotenv.load_dotenv()
    __queue = os.getenv("RABBITMQ_QUEUE_NAME")
    __host = os.getenv("RABBITMQ_HOST")
    __username = os.getenv("RABBITMQ_DEFAULT_USER")
    __password = os.getenv("RABBITMQ_DEFAULT_PASS")
    __port = os.getenv("RABBITMQ_PORT_AMQP")

    @classmethod
    def __get_parameters(cls):
        print(f"queue: {cls.__queue}")
        print(f"host: {cls.__host}")
        print(f"username: {cls.__username}")
        print(f"password: {cls.__password}")  
        print(f"port: {cls.__port}")  

        return pika.ConnectionParameters(
            host=cls.__host,
            credentials=pika.PlainCredentials(
                username=cls.__username,
                password=cls.__password,
            ),
            port=cls.__port
        )

    @classmethod
    def publish(
        cls,
        member_id: str,
        domain: str,
        temp_file_path: str,
        project: str,
        name: str
    ):
        # RabbitMQ 서버에 연결
        with pika.BlockingConnection(cls.__get_parameters()) as connection:
            channel = connection.channel()

            # 큐 생성 (이미 존재하면 무시)
            channel.queue_declare(queue=cls.__queue)
            temp_file_path = temp_file_path.replace("\\\\", "\\")
            # 메시지 전송
            message = json.dumps(
                {
                    "member_id": member_id,
                    "domain": domain,
                    "temp_file_path": temp_file_path,
                    "project": project,
                    "name": name
                }
            ).encode()

            channel.basic_publish(
                exchange="",
                routing_key=cls.__queue,
                body=message
            )
            print(f" [x] Sent {message}")
