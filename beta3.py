#!/usr/bin/env python3
"""
Product Category Classification - Hierarchical Embedding + Cross-Encoder Re-ranking (BETA3)
============================================================================================
Samma som beta.py men med CROSS-ENCODER RE-RANKING för bättre accuracy.

Flöde:
1. Embedding similarity → hämta top-K kandidater (snabbt)
2. Cross-encoder re-ranking → poängsätt varje kandidat (noggrannt)
3. Välj bästa från re-rankade listan

Modeller:
- Embedding: BAAI/bge-m3 (bi-encoder, snabb)
- Re-ranking: BAAI/bge-reranker-v2-m3 (cross-encoder, noggrann)

GPU: Optimerat för A100 40GB VRAM
"""

import asyncio
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import asyncpg
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Konfigurera logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ladda environment variabler
load_dotenv()

# Paths
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"


# =============================================================================
# KONFIGURATION
# =============================================================================

@dataclass
class Config:
    """Konfiguration för A100 40GB VRAM."""
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))

    # Embedding model (bi-encoder - snabb)
    model_name: str = "BAAI/bge-m3"

    # Re-ranker model (cross-encoder - noggrann)
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Antal kandidater att hämta innan re-ranking
    top_k_candidates: int = 10

    batch_size: int = 1024
    rerank_batch_size: int = 64  # Mindre batch för cross-encoder
    max_length: int = 256
    min_confidence: float = 0.3  # Lägre threshold för cross-encoder scores

    # Vikter för Level 1 (product_type + title, ingen description)
    l1_weight_type: float = 0.6
    l1_weight_title: float = 0.4
    l1_weight_desc: float = 0.0

    # Vikter för Level 2+ (alla fält)
    l2_weight_type: float = 0.5
    l2_weight_title: float = 0.3
    l2_weight_desc: float = 0.2

    def get_db_url(self) -> str:
        return self.db_url or ""


# =============================================================================
# CATEGORY CACHE - Ladda en gång, spara lokalt
# =============================================================================

class CategoryCache:
    """Cache kategorier lokalt för att slippa DB-anrop varje gång."""

    # Gender mapping: CSV gender value -> database gender_id
    GENDER_MAP = {
        'male': 1,      # Herr
        'female': 7,    # Dam
        'unisex': None,  # Matcha mot alla
    }

    def __init__(self, cache_file: Path = CACHE_DIR / "categories.json"):
        self.cache_file = cache_file
        self.categories = []
        self.by_level = {}
        self.by_parent = {}
        self.by_id = {}
        self.by_level_gender = {}

    def load_from_cache(self) -> bool:
        """Ladda från lokal cache om den finns."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.categories = json.load(f)
                self._build_indexes()
                logger.info(f"Laddade {len(self.categories)} kategorier från cache")
                return True
            except Exception as e:
                logger.warning(f"Kunde inte ladda cache: {e}")
        return False

    def save_to_cache(self):
        """Spara till lokal cache."""
        self.cache_file.parent.mkdir(exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, ensure_ascii=False, indent=2)
        logger.info(f"Sparade {len(self.categories)} kategorier till cache")

    async def load_from_db(self, db_url: str):
        """Ladda från databas."""
        logger.info("Hämtar kategorier från databas...")
        conn = await asyncpg.connect(db_url)
        try:
            rows = await conn.fetch("""
                SELECT id, name, parent_id, level, path, slug, gender_id
                FROM category
                WHERE active = true AND deleted_at IS NULL
                ORDER BY level, name
            """)
            self.categories = [dict(row) for row in rows]
            self._build_indexes()
            self.save_to_cache()
            logger.info(f"Hämtade {len(self.categories)} kategorier från DB")
        finally:
            await conn.close()

    def _build_indexes(self):
        """Bygg index för snabb lookup."""
        self.by_level = {}
        self.by_parent = {}
        self.by_id = {}
        self.by_level_gender = {}

        for cat in self.categories:
            cat_id = cat['id']
            level = cat.get('level', 1)
            parent_id = cat.get('parent_id')
            gender_id = cat.get('gender_id')

            self.by_id[cat_id] = cat

            if level not in self.by_level:
                self.by_level[level] = []
            self.by_level[level].append(cat)

            key = (level, gender_id)
            if key not in self.by_level_gender:
                self.by_level_gender[key] = []
            self.by_level_gender[key].append(cat)

            if parent_id is not None:
                if parent_id not in self.by_parent:
                    self.by_parent[parent_id] = []
                self.by_parent[parent_id].append(cat)

    def get_level(self, level: int, gender: str = None) -> list[dict]:
        """Hämta kategorier på en viss nivå, eventuellt filtrerat på gender."""
        if gender and gender.lower() in self.GENDER_MAP:
            gender_id = self.GENDER_MAP[gender.lower()]
            if gender_id is not None:
                return self.by_level_gender.get((level, gender_id), [])
        return self.by_level.get(level, [])

    def get_children(self, parent_id: int) -> list[dict]:
        """Hämta alla barn till en kategori."""
        return self.by_parent.get(parent_id, [])

    def get_by_id(self, cat_id: int) -> Optional[dict]:
        """Hämta kategori via ID."""
        return self.by_id.get(cat_id)

    def get_gender_id(self, gender: str) -> Optional[int]:
        """Konvertera CSV gender till database gender_id."""
        if gender and gender.lower() in self.GENDER_MAP:
            return self.GENDER_MAP[gender.lower()]
        return None


# =============================================================================
# EMBEDDING MODEL (Bi-encoder)
# =============================================================================

class EmbeddingModel:
    """Multilingual embedding-modell (bi-encoder)."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.category_embeddings_cache = {}

    def load(self):
        """Ladda modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar embedding-modell: {self.config.model_name}")
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling för sentence embedding."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: list[str], prefix: str = "") -> torch.Tensor:
        """Koda texter till embeddings."""
        self.load()

        if prefix:
            prefixed = [f"{prefix}{t}" for t in texts]
        else:
            prefixed = texts
        all_emb = []

        for i in range(0, len(prefixed), self.config.batch_size):
            batch = prefixed[i:i + self.config.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.config.max_length, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = self._mean_pooling(outputs, inputs['attention_mask'])
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_emb.append(emb)

        return torch.cat(all_emb, dim=0)

    def get_category_embeddings(self, categories: list[dict], cache_key: str,
                                   category_cache: 'CategoryCache' = None) -> tuple[torch.Tensor, list[dict], list[str]]:
        """Hämta/skapa embeddings för kategorier (med cache). Returnerar också namnen."""
        if cache_key in self.category_embeddings_cache:
            return self.category_embeddings_cache[cache_key]

        # Berika kategorier med barn för bättre matching
        if category_cache:
            names = []
            for cat in categories:
                children = category_cache.get_children(cat['id'])
                if children:
                    child_names = [c['name'] for c in children]
                    enriched = f"{cat['name']}: {', '.join(child_names)}"
                    names.append(enriched)
                else:
                    names.append(cat['name'])
            if cache_key.startswith("level_1"):
                logger.info(f"Berikade Level 1 kategorier ({cache_key}): {len(categories)} st")
        else:
            names = [cat['name'] for cat in categories]

        embeddings = self.encode(names)

        self.category_embeddings_cache[cache_key] = (embeddings, categories, names)
        return embeddings, categories, names


# =============================================================================
# CROSS-ENCODER RE-RANKER
# =============================================================================

class CrossEncoderReranker:
    """Cross-encoder för re-ranking av kandidater."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Ladda cross-encoder modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar cross-encoder: {self.config.reranker_model}")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.reranker_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reranker_model,
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Cross-encoder laddad på {self.device}")

    def rerank(self, query: str, candidates: list[str]) -> list[tuple[int, float]]:
        """
        Re-ranka kandidater mot en query.

        Returns: Lista av (original_index, score) sorterad efter score (högst först)
        """
        self.load()

        if not candidates:
            return []

        # Skapa query-kandidat par
        pairs = [[query, cand] for cand in candidates]

        scores = []
        for i in range(0, len(pairs), self.config.rerank_batch_size):
            batch = pairs[i:i + self.config.rerank_batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # BGE reranker returnerar logits, använd sigmoid för score 0-1
                batch_scores = torch.sigmoid(outputs.logits.squeeze(-1))
                scores.extend(batch_scores.cpu().tolist())

        # Om bara en kandidat, scores är en float istället för lista
        if isinstance(scores, float):
            scores = [scores]

        # Returnera (index, score) sorterat efter score
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores

    def rerank_batch(self, queries: list[str], candidates_per_query: list[list[str]]) -> list[list[tuple[int, float]]]:
        """
        Re-ranka kandidater för flera queries samtidigt.

        Returns: Lista av listor med (original_index, score) per query
        """
        results = []
        for query, candidates in zip(queries, candidates_per_query):
            results.append(self.rerank(query, candidates))
        return results


# =============================================================================
# HIERARCHICAL CLASSIFIER WITH RE-RANKING
# =============================================================================

class HierarchicalClassifierWithReranking:
    """
    Hierarkisk kategori-klassificerare MED cross-encoder re-ranking.

    Flöde per nivå:
    1. Embedding similarity → top-K kandidater
    2. Cross-encoder re-ranking → bästa match
    """

    def __init__(self, config: Config, category_cache: CategoryCache,
                 embedding_model: EmbeddingModel, reranker: CrossEncoderReranker):
        self.config = config
        self.cache = category_cache
        self.model = embedding_model
        self.reranker = reranker
        self.max_levels = 4

    def _get_product_fields(self, product: dict) -> tuple[str, str, str, str]:
        """Extrahera produktfält."""
        product_type = (product.get('product_type', '') or '').strip()
        title = (product.get('title', '') or '').strip()
        description = ((product.get('description', '') or '')[:300]).strip()

        # Extrahera de 2 sista delarna av google_product_category
        google_cat = product.get('google_product_category', '') or ''
        if '>' in google_cat:
            parts = [p.strip() for p in google_cat.split('>')]
            google_cat = ' '.join(parts[-2:]) if len(parts) >= 2 else parts[-1]

        return product_type, title, description, google_cat

    def _build_product_text(self, product: dict) -> str:
        """Bygg produkttext för cross-encoder."""
        product_type, title, description, google_cat = self._get_product_fields(product)
        parts = [p for p in [product_type, google_cat, title] if p]
        return ' | '.join(parts) if parts else 'unknown'

    def encode_products_weighted(self, products: list[dict],
                                  weight_type: float = 0.5,
                                  weight_title: float = 0.3,
                                  weight_desc: float = 0.2) -> torch.Tensor:
        """Skapa viktade embeddings för produkter."""
        types = []
        google_cats = []
        titles = []
        for p in products:
            pt, t, d, gc = self._get_product_fields(p)
            types.append(pt if pt else 'unknown')
            google_cats.append(gc if gc else pt if pt else 'unknown')
            titles.append(t if t else 'unknown')

        emb_types = self.model.encode(types)
        emb_google = self.model.encode(google_cats)
        emb_titles = self.model.encode(titles)

        total = weight_type + weight_title + weight_desc
        w_type = (weight_type * 0.35) / total
        w_google = (weight_type * 0.65) / total
        w_title = (weight_title + weight_desc) / total

        combined = (w_type * emb_types +
                   w_google * emb_google +
                   w_title * emb_titles)

        combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        return combined

    def _get_top_k_candidates(self, product_embeddings: torch.Tensor,
                               cat_embeddings: torch.Tensor,
                               categories: list[dict],
                               category_names: list[str],
                               k: int) -> list[list[tuple[dict, str, float]]]:
        """
        Hämta top-K kandidater per produkt baserat på embedding similarity.

        Returns: Lista av listor med (category, enriched_name, similarity_score) per produkt
        """
        if len(categories) == 0:
            return [[] for _ in range(product_embeddings.shape[0])]

        cat_emb = cat_embeddings.to(product_embeddings.device)
        similarities = torch.mm(product_embeddings, cat_emb.T)

        # Hämta top-K per produkt
        actual_k = min(k, len(categories))
        top_scores, top_indices = torch.topk(similarities, actual_k, dim=1)

        results = []
        for i in range(product_embeddings.shape[0]):
            candidates = []
            for j in range(actual_k):
                idx = top_indices[i, j].item()
                score = top_scores[i, j].item()
                candidates.append((categories[idx], category_names[idx], score))
            results.append(candidates)

        return results

    def _match_with_reranking(self, products: list[dict],
                               product_embeddings: torch.Tensor,
                               categories: list[dict],
                               cat_embeddings: torch.Tensor,
                               category_names: list[str]) -> list[tuple[dict, float]]:
        """
        Matcha produkter mot kategorier med re-ranking.

        1. Hämta top-K kandidater via embedding similarity
        2. Re-ranka med cross-encoder
        3. Returnera bästa match
        """
        if len(categories) == 0:
            return [(None, 0.0)] * len(products)

        # Steg 1: Hämta top-K kandidater
        top_k = self._get_top_k_candidates(
            product_embeddings, cat_embeddings, categories, category_names,
            k=self.config.top_k_candidates
        )

        # Steg 2: Re-ranka varje produkts kandidater
        results = []
        for i, product in enumerate(products):
            candidates = top_k[i]
            if not candidates:
                results.append((None, 0.0))
                continue

            # Bygg query-text för denna produkt
            query = self._build_product_text(product)

            # Extrahera kandidat-namn för re-ranking
            candidate_names = [name for (cat, name, emb_score) in candidates]

            # Re-ranka
            reranked = self.reranker.rerank(query, candidate_names)

            if reranked:
                # Bästa kandidaten efter re-ranking
                best_idx, best_score = reranked[0]
                best_cat = candidates[best_idx][0]
                results.append((best_cat, best_score))
            else:
                # Fallback till embedding-score
                best_cat, _, emb_score = candidates[0]
                results.append((best_cat, emb_score))

        return results

    def classify_batch(self, products: list[dict]) -> list[dict]:
        """Klassificera en batch hierarkiskt med re-ranking."""
        results = [{
            'level_1': None, 'level_1_id': None, 'level_1_conf': 0.0,
            'level_2': None, 'level_2_id': None, 'level_2_conf': 0.0,
            'level_3': None, 'level_3_id': None, 'level_3_conf': 0.0,
            'level_4': None, 'level_4_id': None, 'level_4_conf': 0.0,
        } for _ in products]

        # === LEVEL 1 med gender-filtrering och re-ranking ===
        gender_groups = {}
        for i, p in enumerate(products):
            gender = (p.get('gender', '') or '').lower()
            if gender not in gender_groups:
                gender_groups[gender] = []
            gender_groups[gender].append(i)

        # Skapa embeddings för Level 1
        product_emb_l1 = self.encode_products_weighted(
            products,
            weight_type=self.config.l1_weight_type,
            weight_title=self.config.l1_weight_title,
            weight_desc=self.config.l1_weight_desc
        )

        # Debug
        if products:
            p0 = products[0]
            logger.info(f"CSV kolumner: {list(p0.keys())}")
            logger.info(f"Gender-grupper: {list(gender_groups.keys())}")

        # Matcha varje gender-grupp
        for gender, indices in gender_groups.items():
            level1_cats = self.cache.get_level(1, gender if gender else None)
            if not level1_cats:
                level1_cats = self.cache.get_level(1)
            if not level1_cats:
                continue

            cache_key = f"level_1_gender_{gender}" if gender else "level_1"
            l1_emb, l1_cats, l1_names = self.model.get_category_embeddings(level1_cats, cache_key, self.cache)

            # Hämta produkter i denna grupp
            group_products = [products[i] for i in indices]
            group_emb = product_emb_l1[indices]

            # Matcha med re-ranking
            l1_matches = self._match_with_reranking(
                group_products, group_emb, l1_cats, l1_emb, l1_names
            )

            for j, (cat, conf) in enumerate(l1_matches):
                idx = indices[j]
                if cat and conf >= self.config.min_confidence:
                    results[idx]['level_1'] = cat['name']
                    results[idx]['level_1_id'] = cat['id']
                    results[idx]['level_1_conf'] = round(conf, 4)

        # === LEVEL 2, 3, 4 med re-ranking ===
        product_emb_full = self.encode_products_weighted(
            products,
            weight_type=self.config.l2_weight_type,
            weight_title=self.config.l2_weight_title,
            weight_desc=self.config.l2_weight_desc
        )

        for level in [2, 3, 4]:
            prev_level = level - 1
            prev_key = f'level_{prev_level}_id'

            parent_groups = {}
            for i, result in enumerate(results):
                parent_id = result.get(prev_key)
                if parent_id:
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(i)

            for parent_id, indices in parent_groups.items():
                children = self.cache.get_children(parent_id)
                if not children:
                    continue

                cache_key = f"level_{level}_parent_{parent_id}"
                child_emb, child_cats, child_names = self.model.get_category_embeddings(children, cache_key, self.cache)

                group_products = [products[i] for i in indices]
                group_emb = product_emb_full[indices]

                matches = self._match_with_reranking(
                    group_products, group_emb, child_cats, child_emb, child_names
                )

                for j, (cat, conf) in enumerate(matches):
                    idx = indices[j]
                    if cat and conf >= self.config.min_confidence:
                        results[idx][f'level_{level}'] = cat['name']
                        results[idx][f'level_{level}_id'] = cat['id']
                        results[idx][f'level_{level}_conf'] = round(conf, 4)

        return results

    def classify_all(self, products: list[dict]) -> list[dict]:
        """Klassificera alla produkter."""
        all_results = []

        with tqdm(total=len(products), desc="Klassificerar med re-ranking", unit="produkt") as pbar:
            for i in range(0, len(products), self.config.batch_size):
                batch = products[i:i + self.config.batch_size]
                results = self.classify_batch(batch)
                all_results.extend(results)
                pbar.update(len(batch))

        return all_results


# =============================================================================
# CSV HANTERING
# =============================================================================

def read_csv(path: str) -> list[dict]:
    """Läs CSV med samma format som app.py."""
    logger.info(f"Läser CSV: {path}")
    products = []
    with open(path, 'r', encoding='iso-8859-1') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            products.append(row)
    logger.info(f"Laddade {len(products)} produkter")
    return products


def write_csv(products: list[dict], results: list[dict], path: str):
    """Skriv resultat till CSV."""
    logger.info(f"Skriver CSV: {path}")

    if not products:
        return

    output_rows = []
    for product, result in zip(products, results):
        row = dict(product)
        row['level_1_kategori'] = result['level_1'] or ''
        row['level_1_kategori_id'] = result['level_1_id'] or ''
        row['level_1_confidence'] = result['level_1_conf']
        row['level_2_kategori'] = result['level_2'] or ''
        row['level_2_kategori_id'] = result['level_2_id'] or ''
        row['level_2_confidence'] = result['level_2_conf']
        row['level_3_kategori'] = result['level_3'] or ''
        row['level_3_kategori_id'] = result['level_3_id'] or ''
        row['level_3_confidence'] = result['level_3_conf']
        row['level_4_kategori'] = result['level_4'] or ''
        row['level_4_kategori_id'] = result['level_4_id'] or ''
        row['level_4_confidence'] = result['level_4_conf']
        output_rows.append(row)

    fieldnames = list(output_rows[0].keys())
    with open(path, 'w', encoding='iso-8859-1', newline='', errors='replace') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"Sparade {len(output_rows)} rader")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Huvudfunktion."""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarkisk klassificering med cross-encoder re-ranking")
    parser.add_argument("--input", "-i", type=str, default=str(INPUT_DIR / "products.csv"))
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--refresh-cache", action="store_true", help="Tvinga omladdning från DB")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--top-k", "-k", type=int, default=10,
                       help="Antal kandidater att hämta innan re-ranking (default: 10)")
    parser.add_argument("--model", "-m", type=str, default="BAAI/bge-m3",
                       help="Embedding model (default: BAAI/bge-m3)")
    parser.add_argument("--reranker", "-r", type=str, default="BAAI/bge-reranker-v2-m3",
                       help="Cross-encoder reranker model (default: BAAI/bge-reranker-v2-m3)")
    # Vikter
    parser.add_argument("--l1-weight-type", type=float, default=0.6)
    parser.add_argument("--l1-weight-title", type=float, default=0.4)
    parser.add_argument("--l2-weight-type", type=float, default=0.5)
    parser.add_argument("--l2-weight-title", type=float, default=0.3)
    parser.add_argument("--l2-weight-desc", type=float, default=0.2)
    args = parser.parse_args()

    config = Config()
    config.model_name = args.model
    config.reranker_model = args.reranker
    config.batch_size = args.batch_size
    config.top_k_candidates = args.top_k
    config.min_confidence = args.min_confidence
    config.l1_weight_type = args.l1_weight_type
    config.l1_weight_title = args.l1_weight_title
    config.l1_weight_desc = 0.0
    config.l2_weight_type = args.l2_weight_type
    config.l2_weight_title = args.l2_weight_title
    config.l2_weight_desc = args.l2_weight_desc

    logger.info("=" * 60)
    logger.info("HIERARKISK KLASSIFICERING MED CROSS-ENCODER RE-RANKING")
    logger.info("=" * 60)
    logger.info(f"Embedding-modell: {config.model_name}")
    logger.info(f"Re-ranker: {config.reranker_model}")
    logger.info(f"Top-K kandidater: {config.top_k_candidates}")
    logger.info(f"Min confidence: {config.min_confidence}")

    # === Ladda kategorier ===
    category_cache = CategoryCache()

    if not args.refresh_cache and category_cache.load_from_cache():
        pass
    else:
        if not config.get_db_url():
            logger.error("DATABASE_URL saknas!")
            sys.exit(1)
        await category_cache.load_from_db(config.get_db_url())

    for level in range(1, 5):
        cats = category_cache.get_level(level)
        logger.info(f"Level {level}: {len(cats)} kategorier")

    # === Ladda produkter ===
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input-fil hittades inte: {input_path}")
        sys.exit(1)

    products = read_csv(str(input_path))
    if not products:
        logger.error("Inga produkter!")
        sys.exit(1)

    # === Klassificera ===
    embedding_model = EmbeddingModel(config)
    reranker = CrossEncoderReranker(config)
    classifier = HierarchicalClassifierWithReranking(config, category_cache, embedding_model, reranker)

    logger.info("=" * 60)
    logger.info(f"KLASSIFICERAR {len(products)} PRODUKTER")
    logger.info("=" * 60)

    start_time = time.time()
    results = classifier.classify_all(products)
    elapsed = time.time() - start_time

    # === Spara ===
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{input_path.stem}_beta3.csv"
    write_csv(products, results, str(output_path))

    # === Sammanfattning ===
    logger.info("=" * 60)
    logger.info("KLART!")
    logger.info("=" * 60)
    logger.info(f"Produkter: {len(products)}")
    logger.info(f"Tid: {elapsed:.1f}s ({len(products)/elapsed:.0f} produkter/sek)")
    logger.info(f"Output: {output_path}")

    # Visa exempel
    print("\n" + "=" * 60)
    print("EXEMPEL")
    print("=" * 60)
    for i in range(min(5, len(products))):
        p = products[i]
        r = results[i]
        print(f"\n{i+1}. {p.get('title', '')[:50]}...")
        print(f"   L1: {r['level_1']} ({r['level_1_conf']:.0%})")
        print(f"   L2: {r['level_2']} ({r['level_2_conf']:.0%})")
        print(f"   L3: {r['level_3']} ({r['level_3_conf']:.0%})")
        print(f"   L4: {r['level_4']} ({r['level_4_conf']:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
