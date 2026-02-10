import argparse
import asyncio
import json
import os
import sys
from typing import List, Dict, Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from ragas.metrics.collections import ContextRecall, ContextPrecision
from ragas.llms import llm_factory
from ragas import experiment

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.RAG.rag import Rag, RagConfig
from src.RAG.hybrid_retriever import HybridRetriever, RetrieverConfig

DEFAULT_CONFIG = {
    "testset_path": "data/singlehop_testset.json",
    "chroma_db_path": "db/chroma_db",
    "bm25_path": "db/bm25",
    "collection_name": "test",
    "embedding_model": "solar-embedding-1-large",
    "llm_model": "gpt-4o-mini",
    "retrieval_k": 5
}


def load_eval_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[RAG][WARN] config not found: {path}. Using defaults.")
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    eval_cfg = raw.get("evaluation", {})
    merged = DEFAULT_CONFIG.copy()
    merged.update(eval_cfg)
    return merged


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/rag_metric.yaml",
        help="e.g. config/rag_metric.yaml",
    )
    args = parser.parse_args()
    cfg = load_eval_config(args.config)

    load_dotenv()
    
    async_client = AsyncOpenAI()
    llm = llm_factory(cfg["llm_model"], client=async_client)

    bm25_path = cfg["bm25_path"]
    db_path = cfg["chroma_db_path"]
    ragconfig = RagConfig(
        db_path=db_path,
        bm25_path=bm25_path,
        embedding_model_name=cfg["embedding_model"],
        collection_name=cfg["collection_name"],
    )
    rag = Rag(config=ragconfig)
    db = rag.load()
    config = RetrieverConfig(db=db, pickle_path=bm25_path, top_k=cfg["retrieval_k"])
    retriever = HybridRetriever(config=config)

    if not os.path.exists(cfg["testset_path"]):
        print(f"❌ 파일을 찾을 수 없습니다: {cfg['testset_path']}")
        return

    with open(cfg["testset_path"], 'r', encoding='utf-8') as f:
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
        retrieved_docs = retriever.retrieve(query)
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
