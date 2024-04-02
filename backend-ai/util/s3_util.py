import boto3
from boto3.s3.transfer import TransferConfig
from dotenv import load_dotenv
import os

from typing import Dict, Any


class S3Util:
    load_dotenv()
    __bucket = os.getenv("S3_BUCKET_NAME")
    __access_key = os.getenv("S3_ACCESS_KEY")
    __secret_key = os.getenv("S3_SECRET_KEY")
    __region_name = os.getenv("S3_REGION_NAME")

    __s3 = boto3.client(
        service_name="s3",
        region_name=__region_name,
        aws_access_key_id=__access_key,
        aws_secret_access_key=__secret_key
    )

        # S3 파일 전송 configuration
    __config: TransferConfig = TransferConfig(
        multipart_threshold=1024 * 25,  # 25 MB
        max_concurrency=10,  # 최대 동시 업로드/다운로드 개수
        multipart_chunksize=1024 * 25,  # 각 멀티파트의 크기 25 MB
        use_threads=True  # 멀티 스레드 사용
    )


    @classmethod
    def upload_obj_to_s3(
            cls,
            src: str,
            dest: str # S3://__bucket/dest
    ):
        with open(file=src, mode="rb") as fileobj:
            cls.__s3.upload_fileobj(
                Fileobj=fileobj,
                Key=dest,
                Bucket=cls.__bucket,
                Config=cls.__config
            )

        url = f"https://{cls.__bucket}.s3.ap-northeast-2.amazonaws.com/{dest}"

        return url


    @classmethod
    def upload_to_s3(
            cls,
            src: str,
            dest: str
    ) -> str:
        cls.__s3.upload_file(
            Filename=src, # 여기에는 파일 업로드 src 경로
            Bucket=cls.__bucket,
            Key=dest # 여기에는 파일 업로드 dest 경로
        )

        url = f"https://{cls.__bucket}.s3.ap-northeast-2.amazonaws.com/{dest}"

        return url


    @classmethod
    def delete_from_s3(
            cls,
            path: str
    ):
        '''
        {
            'ResponseMetadata': {
                'RequestId': 'AEQE16J3TVY5Q779',
                'HostId': 'nZ022adze2V3IsAF71c1uNDAkPzjS8dXefvphaQfHU4kx0/wJAEzimypPzYEojaeurPka9oSRLM=',
                'HTTPStatusCode': 204,
                'HTTPHeaders': {
                    'x-amz-id-2': 'nZ022adze2V3IsAF71c1uNDAkPzjS8dXefvphaQfHU4kx0/wJAEzimypPzYEojaeurPka9oSRLM=',
                    'x-amz-request-id': 'AEQE16J3TVY5Q779',
                    'date': 'Wed, 27 Mar 2024 06:05:37 GMT',
                    'server': 'AmazonS3'
                },
                'RetryAttempts': 0
            }
        }

        '''

        # 없는 파일 지워도 성공이 반환된다.
        res: Dict[str, Any] = cls.__s3.delete_object(
            Bucket=cls.__bucket,
            Key=path
        )

        # res_status_code = res["ResponseMetadata"]["HTTPStatusCode"]
        return res

    @classmethod
    def move(cls, old_prefix, new_prefix):
        src_objs = cls.__list_objects_with_prefix(old_prefix)
        print(f"src_objs: {src_objs}")
        
        for src_obj in src_objs:
            src_key: str = src_obj["Key"]
            print(f"===============================src_key: {src_key}")
            dest_key = src_key.replace(old_prefix, new_prefix, 1) # 1 = 한 번만 변경한다

            # 복사
            cls.__s3.copy_object(
                Bucket=cls.__bucket,
                CopySource={
                    "Bucket": cls.__bucket,
                    "Key": src_key},
                Key=dest_key)

            # 원본 삭제
            cls.__s3.delete_object(
                Bucket=cls.__bucket,
                Key=src_key
            )


    @classmethod
    def __list_objects_with_prefix(cls, prefix):
        response = cls.__s3.list_objects_v2(
            Bucket=cls.__bucket,
            Prefix=prefix
        )
        
        return response.get('Contents', [])
