import logging
from abc import ABC, abstractmethod

import qianfan
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(
    format="%(asctime)s - %(pathname)s - %(message)s",
    level=logging.INFO
)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class EBEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="Embedding-V1"):
        self.client = qianfan.Embedding()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.do(texts=[text], model=self.model)["body"]["data"][0]["embedding"]
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)
