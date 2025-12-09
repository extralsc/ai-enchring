#!/usr/bin/env python3
"""
Product Category Classification - FashionSigLIP (FASHION BETA)
==============================================================
Samma som beta.py men använder Marqo/marqo-fashionSigLIP istället för BGE-M3.

FashionSigLIP är tränad på fashion-data och förstår:
- Klädesplagg (jackets, sweaters, dresses, etc.)
- Material (wool, cotton, leather, etc.)
- Stilar (casual, formal, sporty, etc.)

Använder CLIP:s text-encoder för både produkter och kategorier.
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

    # FashionSigLIP - tränad på fashion data
    model_name: str = "Marqo/marqo-fashionSigLIP"

    batch_size: int = 512  # Mindre batch för CLIP
    max_length: int = 77  # CLIP max token length
    min_confidence: float = 0.15  # CLIP scores är lägre än embedding models

    # Vikter för Level 1
    l1_weight_type: float = 0.6
    l1_weight_title: float = 0.4
    l1_weight_desc: float = 0.0

    # Vikter för Level 2+
    l2_weight_type: float = 0.5
    l2_weight_title: float = 0.3
    l2_weight_desc: float = 0.2

    def get_db_url(self) -> str:
        return self.db_url or ""


# =============================================================================
# CATEGORY CACHE
# =============================================================================

class CategoryCache:
    """Cache kategorier lokalt för att slippa DB-anrop varje gång."""

    GENDER_MAP = {
        'male': 1,
        'female': 7,
        'unisex': None,
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
        """Hämta kategorier på en viss nivå."""
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


# =============================================================================
# FASHIONSIGLIP MODEL
# =============================================================================

class FashionSigLIPModel:
    """
    FashionSigLIP för fashion-specifik text embedding.

    Använder CLIP:s text-encoder som är tränad på fashion-data.
    Förstår: jackets, sweaters, hoodies, materials, styles, etc.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.category_embeddings_cache = {}

    def load(self):
        """Ladda FashionSigLIP modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar FashionSigLIP: {self.config.model_name}")
        import open_clip

        # Ladda modell via open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f'hf-hub:{self.config.model_name}'
        )
        self.tokenizer = open_clip.get_tokenizer(f'hf-hub:{self.config.model_name}')

        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        logger.info("FashionSigLIP laddad!")

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Koda texter till embeddings med CLIP text encoder."""
        self.load()

        all_emb = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]

            # Tokenize
            tokens = self.tokenizer(batch).to(self.device)

            with torch.no_grad():
                # Encode text
                text_features = self.model.encode_text(tokens)
                # Normalize
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_emb.append(text_features)

        return torch.cat(all_emb, dim=0)

    def get_category_embeddings(self, categories: list[dict], cache_key: str,
                                   category_cache: 'CategoryCache' = None) -> tuple[torch.Tensor, list[dict]]:
        """Hämta/skapa embeddings för kategorier (med cache)."""
        if cache_key in self.category_embeddings_cache:
            return self.category_embeddings_cache[cache_key]

        # Berika kategorier med barn + fashion-kontext
        if category_cache:
            names = []
            for cat in categories:
                children = category_cache.get_children(cat['id'])
                if children:
                    child_names = [c['name'] for c in children]
                    # Lägg till "clothing category" för bättre CLIP-förståelse
                    enriched = f"{cat['name']} clothing: {', '.join(child_names)}"
                    names.append(enriched)
                else:
                    # Lägg till kontext för CLIP
                    names.append(f"{cat['name']} fashion category")
            if cache_key.startswith("level_1"):
                logger.info(f"Berikade Level 1 kategorier ({cache_key}): {len(categories)} st")
                for n in names[:3]:
                    logger.info(f"  -> {n[:80]}...")
        else:
            names = [f"{cat['name']} fashion category" for cat in categories]

        embeddings = self.encode_text(names)

        self.category_embeddings_cache[cache_key] = (embeddings, categories)
        return embeddings, categories


# =============================================================================
# HIERARCHICAL CLASSIFIER
# =============================================================================

class HierarchicalClassifier:
    """
    Hierarkisk kategori-klassificerare med FashionSigLIP.
    """

    def __init__(self, config: Config, category_cache: CategoryCache, model: FashionSigLIPModel):
        self.config = config
        self.cache = category_cache
        self.model = model
        self.max_levels = 4

    def _get_product_fields(self, product: dict) -> tuple[str, str, str, str]:
        """Extrahera produktfält."""
        product_type = (product.get('product_type', '') or '').strip()
        title = (product.get('title', '') or '').strip()
        description = ((product.get('description', '') or '')[:200]).strip()

        # Extrahera de 2 sista delarna av google_product_category
        google_cat = product.get('google_product_category', '') or ''
        if '>' in google_cat:
            parts = [p.strip() for p in google_cat.split('>')]
            google_cat = ' '.join(parts[-2:]) if len(parts) >= 2 else parts[-1]

        return product_type, title, description, google_cat

    def _build_product_text(self, product: dict, level: int = 1) -> str:
        """Bygg produkttext för CLIP embedding."""
        product_type, title, description, google_cat = self._get_product_fields(product)

        # CLIP fungerar bäst med naturligt språk
        if level == 1:
            # Fokusera på typ och google-kategori för Level 1
            parts = []
            if product_type:
                parts.append(product_type)
            if google_cat:
                parts.append(google_cat)
            if title:
                # Ta bara produktnamnet (utan storlek/färg)
                clean_title = title.split(',')[0].strip()
                parts.append(clean_title)
            return ' '.join(parts) if parts else 'clothing item'
        else:
            parts = [p for p in [product_type, google_cat, title] if p]
            return ' '.join(parts) if parts else 'clothing item'

    def encode_products(self, products: list[dict], level: int = 1) -> torch.Tensor:
        """Skapa embeddings för produkter."""
        texts = [self._build_product_text(p, level) for p in products]
        return self.model.encode_text(texts)

    def _match_to_categories(self, product_embeddings: torch.Tensor, categories: list[dict],
                             cat_embeddings: torch.Tensor) -> list[tuple[dict, float]]:
        """Matcha produkter mot kategorier."""
        if len(categories) == 0:
            return [(None, 0.0)] * product_embeddings.shape[0]

        cat_emb = cat_embeddings.to(product_embeddings.device)

        # Cosine similarity (embeddings är redan normaliserade)
        similarities = product_embeddings @ cat_emb.T

        # CLIP ger scores mellan -1 och 1, skala till 0-1
        similarities = (similarities + 1) / 2

        best_scores, best_indices = torch.max(similarities, dim=1)

        results = []
        for score, idx in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
            results.append((categories[idx], float(score)))

        return results

    def classify_batch(self, products: list[dict]) -> list[dict]:
        """Klassificera en batch hierarkiskt."""
        results = [{
            'level_1': None, 'level_1_id': None, 'level_1_conf': 0.0,
            'level_2': None, 'level_2_id': None, 'level_2_conf': 0.0,
            'level_3': None, 'level_3_id': None, 'level_3_conf': 0.0,
            'level_4': None, 'level_4_id': None, 'level_4_conf': 0.0,
        } for _ in products]

        # === LEVEL 1 med gender-filtrering ===
        gender_groups = {}
        for i, p in enumerate(products):
            gender = (p.get('gender', '') or '').lower()
            if gender not in gender_groups:
                gender_groups[gender] = []
            gender_groups[gender].append(i)

        # Skapa embeddings för Level 1
        product_emb_l1 = self.encode_products(products, level=1)

        # Debug
        if products:
            logger.info(f"Gender-grupper: {list(gender_groups.keys())}")
            # Visa första produktens text
            sample_text = self._build_product_text(products[0], level=1)
            logger.info(f"Exempel produkttext: '{sample_text}'")

        # Matcha varje gender-grupp
        for gender, indices in gender_groups.items():
            level1_cats = self.cache.get_level(1, gender if gender else None)
            if not level1_cats:
                level1_cats = self.cache.get_level(1)
            if not level1_cats:
                continue

            cache_key = f"level_1_gender_{gender}" if gender else "level_1"
            l1_emb, l1_cats = self.model.get_category_embeddings(level1_cats, cache_key, self.cache)

            group_emb = product_emb_l1[indices]
            l1_matches = self._match_to_categories(group_emb, l1_cats, l1_emb)

            for j, (cat, conf) in enumerate(l1_matches):
                idx = indices[j]
                if cat and conf >= self.config.min_confidence:
                    results[idx]['level_1'] = cat['name']
                    results[idx]['level_1_id'] = cat['id']
                    results[idx]['level_1_conf'] = round(conf, 4)

        # === LEVEL 2, 3, 4 ===
        product_emb_full = self.encode_products(products, level=2)

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
                child_emb, child_cats = self.model.get_category_embeddings(children, cache_key, self.cache)

                group_emb = product_emb_full[indices]
                matches = self._match_to_categories(group_emb, child_cats, child_emb)

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

        with tqdm(total=len(products), desc="Klassificerar med FashionSigLIP", unit="produkt") as pbar:
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
    """Läs CSV."""
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

    parser = argparse.ArgumentParser(description="Hierarkisk klassificering med FashionSigLIP")
    parser.add_argument("--input", "-i", type=str, default=str(INPUT_DIR / "products.csv"))
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=512)
    parser.add_argument("--refresh-cache", action="store_true", help="Tvinga omladdning från DB")
    parser.add_argument("--min-confidence", type=float, default=0.15,
                       help="Min confidence (CLIP scores är lägre, default: 0.15)")
    args = parser.parse_args()

    config = Config()
    config.batch_size = args.batch_size
    config.min_confidence = args.min_confidence

    logger.info("=" * 60)
    logger.info("HIERARKISK KLASSIFICERING MED FASHIONSIGLIP")
    logger.info("=" * 60)
    logger.info(f"Modell: {config.model_name}")
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
    model = FashionSigLIPModel(config)
    classifier = HierarchicalClassifier(config, category_cache, model)

    logger.info("=" * 60)
    logger.info(f"KLASSIFICERAR {len(products)} PRODUKTER")
    logger.info("=" * 60)

    start_time = time.time()
    results = classifier.classify_all(products)
    elapsed = time.time() - start_time

    # === Spara ===
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{input_path.stem}_fashion.csv"
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
