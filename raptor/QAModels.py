import logging
import os

import qianfan

import getpass
from abc import ABC, abstractmethod

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5ForConditionalGeneration, T5Tokenizer


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context, question):
        pass


class EB4QAModel(BaseQAModel):
    def __init__(self, model="ERNIE-4.0-8K"):
        """
        Initializes the EB-4 model with the specified model version.

        Args:
            model (str, optional): The EB-4 model version to use for generating summaries. Defaults to "ERNIE-4.0-8K".
        """
        self.model = model
        self.client = qianfan.ChatCompletion()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.do(
                messages=[{
                    "role": "user",
                    "content": f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
                }],
                model=self.model,
            )
            return response.body["result"]

        except Exception as e:
            print(e)
            return ""


class EB3QAModel(BaseQAModel):
    def __init__(self, model="ERNIE-3.5-8K"):
        """
        Initializes the ERNIE-3.5-8K model with the specified model version.

        Args:
            model (str, optional): The EB-3.5 model version to use for generating summaries. Defaults to "ERNIE-3.5-8K".
        """
        self.model = model
        self.client = qianfan.ChatCompletion()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.do(
                messages=[
                    # {"role": "system", "content": "You are Question Answering Portal"},
                    {
                        "role": "user",
                        "content": f"您是解决问题的专家，使用以下的参考信息{context}，回答用户的问题{question}"
                        # "content": f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
                    }
                ],
                model=self.model,
            )
            return response.body["result"]

        except Exception as e:
            print(e)
            return ""

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class EB4TurboQAModel(BaseQAModel):
    def __init__(self, model="ERNIE-4.0-Turbo-8K"):
        """
        Initializes the EB-4 Turbo model with the specified model version.

        Args:
            model (str, optional): The EB-4 Turbo model version to use for generating summaries. Defaults to "ERNIE-4.0-Turbo-8K".
        """
        self.model = model
        self.client = qianfan.ChatCompletion()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(
        self, context, question, max_tokens=150, stop_sequence=None
    ):
        """
        Generates a summary of the given context using the GPT-3 model.

        Args:
            context (str): The text to summarize.
            max_tokens (int, optional): The maximum number of tokens in the generated summary. Defaults to 150.
            stop_sequence (str, optional): The sequence at which to stop summarization. Defaults to None.

        Returns:
            str: The generated summary.
        """
        try:
            response = self.client.do(
                messages=[
                    # {"role": "system", "content": "You are Question Answering Portal"},
                    {
                        "role": "user",
                        "content": f"您是解决问题的专家，使用以下的参考信息{context}，回答用户的问题{question}"
                        # "content": f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible: {question}"
                    }
                ],
                model=self.model,
            )
            return response.body["result"]

        except Exception as e:
            print(e)
            return ""

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, max_tokens=150, stop_sequence=None):

        try:
            return self._attempt_answer_question(
                context, question, max_tokens=max_tokens, stop_sequence=stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name="allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]
