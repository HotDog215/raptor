import logging
import os
from abc import ABC, abstractmethod
import qianfan
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(
    format="%(asctime)s - %(pathname)s - %(message)s",
    level=logging.INFO
)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass


class EB4TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self, model="ERNIE-4.0-Turbo-8K"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = qianfan.ChatCompletion()
            response = client.do(
                messages=[
                    {
                        "role": "user",
                        "content": f"你是个乐于助人的助手。请输出以下内容的总结，总结不超过50个字，包括尽可能多的关键细节：{context}:",
                    }
                ],
                model=self.model,
            )
            return response.body["result"]

        except Exception as e:
            print(e)
            return e


class EB4SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="ERNIE-4.0-8K"):

        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):

        try:
            client = qianfan.ChatCompletion()
            response = client.do(
                messages=[
                    {
                        "role": "user",
                        "content": f"你是个乐于助人的助手。请输出以下内容的总结，总结不超过50个字，包括尽可能多的关键细节：{context}:",
                    }
                ],
                model=self.model,
            )
            return response.body["result"]

        except Exception as e:
            print(e)
            return e
