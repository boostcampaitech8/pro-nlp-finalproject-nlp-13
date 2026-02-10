from __future__ import annotations

import argparse
import asyncio
import os
from typing import List, Dict, Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import QueryDistribution
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

DEFAULT_CONFIG = {
    "data_paths": {
        "guidebook": "data/guidebook/busan_rag_data.json",
        "sports": "data/crawling/sports_crawling.csv",
        "stay": "data/crawling/stay_crawling.csv",
    },
    "output_path": "data/singlehop_testset.json",
    "testset_size": 100,
}


def load_testdata_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[RAG][WARN] config not found: {path}. Using defaults.")
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    test_cfg = raw.get("testdata", {})
    merged = DEFAULT_CONFIG.copy()
    merged.update(test_cfg)
    merged["data_paths"] = {**DEFAULT_CONFIG["data_paths"], **test_cfg.get("data_paths", {})}
    return merged


async def run_generator(config: Dict[str, Any]) -> None:
    load_dotenv()
    
    openai_client = OpenAI() 
    llm = llm_factory("gpt-4o-mini", client=openai_client)
    embeddings = OpenAIEmbeddings(client=openai_client)

    kg = load_data_to_kg(config["data_paths"])

    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=get_personas()
    )        

    query_dist = [
        (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="keyphrases"), 0.5),
        (SingleHopSpecificQuerySynthesizer(llm=llm, property_name="headlines"), 0.5)
    ]
    for query, _ in query_dist:
        prompts = await query.adapt_prompts("korean", llm=llm)
        query.set_prompts(**prompts)
        
    testset = generator.generate(
        testset_size=config["testset_size"],
        query_distribution=query_dist,
    )
  
    df = testset.to_pandas()
    output_path = config["output_path"]
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_json(output_path, orient="records", force_ascii=False, indent=4)
    print(f"테스트셋이 {output_path}으로 저장되었습니다!")


def load_data_to_kg(data_paths: Dict[str, str]) -> KnowledgeGraph:
    kg = KnowledgeGraph()

    guide_df = pd.read_json(data_paths["guidebook"])
    sports_df = pd.read_csv(data_paths["sports"])
    stay_df = pd.read_csv(data_paths["stay"])

    df = pd.concat([guide_df, sports_df, stay_df], ignore_index=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="노드 생성 중"):
        valid_lines = []
        for col_name in df.columns:
            val = row.get(col_name)
            
            if pd.notna(val) and str(val).strip() != "":
                valid_lines.append(f"{col_name}: {val}")

        content = "\n".join(valid_lines)
        if content:
            node = Node(
                type=NodeType.CHUNK,
                properties={
                    "page_content": content,
                    "keyphrases": [str(row.get(k, '')) for k in ['장소명', '구분', '카테고리'] if pd.notna(row.get(k))],
                    "headlines": [str(row.get('장소명', ''))],
                    "document_metadata": {"구분": row.get('구분', '')}
                }
            )
            kg.nodes.append(node)

    print(f"총 {len(kg.nodes)}개의 노드가 통합 로직으로 생성되었습니다.")

    return kg


def get_personas() -> List[Persona]:
    return [
        Persona(
            name="Backpacking College Student",
            role_description="20대 대학생, 대중교통 이용, 가성비 로컬 맛집 및 액티비티 선호."
        ),
        Persona(
            name="Wellness Traveler",
            role_description="30대 직장인, 조용하고 세련된 장소 선호, 감성 숙소 및 우드파이어 식당 관심."
        ),
        Persona(
            name="Family Vacationer",
            role_description="40대 부모, 가족 동반, 노포 맛집 신뢰, 주차 및 안전 최우선."
        )
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/rag_metric.yaml",
        help="e.g. config/rag_metric.yaml",
    )
    args = parser.parse_args()
    cfg = load_testdata_config(args.config)
    asyncio.run(run_generator(cfg))
