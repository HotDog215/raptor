# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @Time    :   2024/7/16 20:05
    @Author  :   ShangZhengYu 
    @Desc    :   注解
"""

__author__ = 'ShangZhengYu'

import logging
import os
from raptor import RetrievalAugmentation
from raptor import RapidOCRDocLoader
os.environ["QIANFAN_ACCESS_KEY"] = "ak"
os.environ["QIANFAN_SECRET_KEY"] = "sk"

logging.basicConfig(
    format="%(asctime)s - %(pathname)s - %(message)s",
    level=logging.INFO
)


if __name__ == "__main__":
    RA = RetrievalAugmentation()
    loader = RapidOCRDocLoader(
        file_path="/Users/shangzhengyu/Downloads/京哈badcase/数字交通“十四五”发展规划.docx"
    )
    docs = loader.load()
    content = docs[0].page_content
    print(f"Content: ", content)
    RA.add_documents(content)

    question = "目前我国数字交通建设主要任务中的「五网」具体是指哪些？"
    print(f"Question: ", question)
    answer = RA.answer_question(question=question)
    print("Answer: ", answer)
