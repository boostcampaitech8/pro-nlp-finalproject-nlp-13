import os
import json
import asyncio
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from datasets import Dataset
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from ragas.metrics.collections import ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas import experiment

CONFIG = {
    "testset_path": "/Users/hanjiseok/pro-nlp-finalproject-nlp-13/singlehop_testset_1.json",
    "chroma_db_path": "/Users/hanjiseok/pro-nlp-finalproject-nlp-13/notebooks/Han/rag_metric/vectorDB/chroma_db",
    "collection_name": "test",
    "embedding_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "retrieval_k": 5
}


async def main():
    load_dotenv()
    
    async_client = AsyncOpenAI()
    llm = llm_factory(CONFIG["llm_model"], client=async_client)
    embeddings = OpenAIEmbeddings(model=CONFIG["embedding_model"])

    vector_db = Chroma(
        persist_directory=CONFIG["chroma_db_path"],
        embedding_function=embeddings,
        collection_name=CONFIG["collection_name"]
    )
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["retrieval_k"]}
    )

    if not os.path.exists(CONFIG["testset_path"]):
        print(f"❌ 파일을 찾을 수 없습니다: {CONFIG['testset_path']}")
        return

    with open(CONFIG["testset_path"], 'r', encoding='utf-8') as f:
        testset_dict = json.load(f)

    evaluation_samples = prepare_rag_dataset(testset_dict, retriever)

    print("Ragas 평가 시작 (Batch Evaluate)...")
    tasks = [run_evaluation(sample, llm) for sample in evaluation_samples]
    results = await asyncio.gather(*tasks)

    if results:
        avg_recall = sum(r.recall_result for r in results) / len(results)
        avg_precision = sum(r.precision_result for r in results) / len(results)

        print("\n" + "="*30)
        print(f"RAG Evaluation Summary")
        print(f"- Total Samples: {len(results)}")
        print(f"- Average Recall: {avg_recall:.4f}")
        print(f"- Average Precision: {avg_precision:.4f}")
        print("="*30)
        
class EvaluationResult(BaseModel):
    recall_result: float
    precision_result: float

@experiment(EvaluationResult)
async def run_evaluation(row: Dict[str, Any], llm: Any) -> EvaluationResult:
    """단일 행에 대한 RAG 평가지표 계산"""
    context_recall = ContextRecall(llm=llm)
    context_precision = ContextPrecision(llm=llm)

    # 비동기로 Recall과 Precision 동시 계산
    recall_task = context_recall.ascore(
        user_input=row['question'],
        reference=row['reference'],
        retrieved_contexts=row['retrieved_contexts']
    )
    precision_task = context_precision.ascore(
        user_input=row['question'],
        reference=row['reference'],
        retrieved_contexts=row['retrieved_contexts']
    )

    recall_score, precision_score = await asyncio.gather(recall_task, precision_task)

    return EvaluationResult(
        recall_result=recall_score,
        precision_result=precision_score
    )


def prepare_rag_dataset(samples: List[Dict], retriever: Any) -> List[Dict]:
    processed_data = []
    
    print(f"{len(samples)}개의 샘플에 대해 Retrieval 수행 중...")
    for sample in samples:
        query = sample["user_input"]
        
        # 검색 수행
        retrieved_docs = retriever.invoke(query)
        retrieved_contexts = [doc.page_content for doc in retrieved_docs]
        
        processed_data.append({
            "question": query,
            "retrieved_contexts": retrieved_contexts,
            "reference": sample.get("reference", ""),
            "reference_contexts": sample.get("reference_contexts", [])
        })
        
    return processed_data

if __name__ == "__main__":
    asyncio.run(main())
