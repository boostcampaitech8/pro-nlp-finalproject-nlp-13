from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import wikipedia
import wikipediaapi


@dataclass(frozen=True)
class WikiConfig:
    lang: str = "ko"
    top_k: int = 5
    max_length: int = 20000
    user_agent: str = "MyApp/1.0 (sucruba7@email.com)"


@dataclass(frozen=True)
class WikiResult:
    title: str
    text: str


class WikipediaRetriever:
    def __init__(
            self,
            cfg: WikiConfig
    ):
        self.cfg = cfg
        wikipedia.set_lang(cfg.lang)
        self.wiki_api = wikipediaapi.Wikipedia(
            language=cfg.lang,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent=cfg.user_agent,
        )
    
    def search_titles(self, keyword: str) -> List[str]:
        try:
            return wikipedia.search(keyword, results=self.cfg.top_k) or []
        except Exception:
            return []

    def get_wiki_text(self, keyword: str) -> Optional[WikiResult]:
        titles = self.search_titles(keyword)
        if not titles:
            return None
        
        for title in titles:
            try:
                page = self.wiki_api.page(title)
                if not page.exists():
                    continue

                text = page.text or ""
                if not text:
                    continue

                if len(text) > self.cfg.max_length:
                    text = text[: self.cfg.max_length] + "\n\n[문서가 길어 앞부분만 제공됨]"

                return WikiResult(title=title, text=text)

            except Exception:
                continue

        return None