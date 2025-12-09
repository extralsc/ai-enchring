#!/usr/bin/env python3
"""
Product Category Classification - Hierarchical Embedding (BETA)
================================================================
Klassificerar produkter till svenska kategorier med HIERARKISK matching:

Steg 1: Matcha mot Level 1 kategorier (t.ex. Kläder, Skor, Accessoarer)
Steg 2: Matcha mot Level 2 under vald Level 1
Steg 3: Matcha mot Level 3 under vald Level 2
Steg 4: Matcha mot Level 4 under vald Level 3 (om finns)

Output: level_1, level_2, level_3, level_4 kolumner

Modell: intfloat/multilingual-e5-large
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
    model_name: str = "BAAI/bge-m3"  # Bättre cross-lingual än E5
    batch_size: int = 1024
    max_length: int = 256
    min_confidence: float = 0.5  # Minimum confidence för match

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
    # Uppdatera dessa baserat på din product_gender tabell
    GENDER_MAP = {
        'male': 1,      # Herr
        'female': 7,    # Dam
        'unisex': None,  # Matcha mot alla
    }

    def __init__(self, cache_file: Path = CACHE_DIR / "categories.json"):
        self.cache_file = cache_file
        self.categories = []
        self.by_level = {}  # {level: [categories]}
        self.by_parent = {}  # {parent_id: [categories]}
        self.by_id = {}  # {id: category}
        self.by_level_gender = {}  # {(level, gender_id): [categories]}

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

            # Index by level + gender
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
# EMBEDDING MODEL
# =============================================================================

class EmbeddingModel:
    """Multilingual embedding-modell."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Cache för kategori-embeddings per nivå
        self.category_embeddings_cache = {}

    def load(self):
        """Ladda modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar modell: {self.config.model_name}")
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

        # BGE-M3 behöver inga prefixes, E5 behöver "query: " / "passage: "
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
                                   category_cache: 'CategoryCache' = None) -> tuple[torch.Tensor, list[dict]]:
        """Hämta/skapa embeddings för kategorier (med cache)."""
        if cache_key in self.category_embeddings_cache:
            return self.category_embeddings_cache[cache_key]

        # Berika ALLA kategorier med barn för bättre matching (skalbart)
        if category_cache:
            names = []
            for cat in categories:
                children = category_cache.get_children(cat['id'])
                if children:
                    child_names = [c['name'] for c in children[:8]]  # Max 8 barn
                    enriched = f"{cat['name']}: {', '.join(child_names)}"
                    names.append(enriched)
                else:
                    names.append(cat['name'])
            if cache_key.startswith("level_1"):
                logger.info(f"Berikade kategorier ({cache_key}, {len(categories)} st):")
                for n in names[:5]:
                    logger.info(f"  -> {n[:100]}...")
        else:
            names = [cat['name'] for cat in categories]

        embeddings = self.encode(names)  # Ingen prefix för BGE-M3

        self.category_embeddings_cache[cache_key] = (embeddings, categories)
        return embeddings, categories


# =============================================================================
# HIERARCHICAL CLASSIFIER
# =============================================================================

class HierarchicalClassifier:
    """
    Hierarkisk kategori-klassificerare.

    Matchar stegvis: Level 1 → Level 2 → Level 3 → Level 4
    """

    def __init__(self, config: Config, category_cache: CategoryCache, embedding_model: EmbeddingModel):
        self.config = config
        self.cache = category_cache
        self.model = embedding_model
        self.max_levels = 4

    def _get_product_fields(self, product: dict) -> tuple[str, str, str, str]:
        """Extrahera produktfält."""
        product_type = (product.get('product_type', '') or '').strip()
        title = (product.get('title', '') or '').strip()
        description = ((product.get('description', '') or '')[:300]).strip()

        # Extrahera mest specifika delen av google_product_category
        # "Apparel & Accessories > Clothing > Sweaters & Hoodies" -> "Sweaters & Hoodies"
        google_cat = product.get('google_product_category', '') or ''
        if '>' in google_cat:
            google_cat = google_cat.split('>')[-1].strip()

        return product_type, title, description, google_cat

    def _build_product_text(self, product: dict, level: int = 1) -> str:
        """Bygg produkttext för embedding (fallback om weighted inte används)."""
        product_type, title, description, google_cat = self._get_product_fields(product)

        if level == 1:
            parts = [p for p in [product_type, google_cat, title] if p]
            return ' | '.join(parts) if parts else 'unknown'
        else:
            parts = [p for p in [product_type, google_cat, title] if p]  # Skip description
            return ' | '.join(parts) if parts else 'unknown'

    def encode_products_weighted(self, products: list[dict],
                                  weight_type: float = 0.5,
                                  weight_title: float = 0.3,
                                  weight_desc: float = 0.2) -> torch.Tensor:
        """
        Skapa viktade embeddings för produkter.

        Använder: product_type, google_product_category, title
        Skippar description (orsakar brus).
        """
        # Extrahera fält
        types = []
        google_cats = []
        titles = []
        for p in products:
            pt, t, d, gc = self._get_product_fields(p)
            types.append(pt if pt else 'unknown')
            google_cats.append(gc if gc else pt if pt else 'unknown')  # Fallback till product_type
            titles.append(t if t else 'unknown')

        # Skapa separata embeddings
        emb_types = self.model.encode(types)
        emb_google = self.model.encode(google_cats)
        emb_titles = self.model.encode(titles)

        # Viktad kombination: type + google_cat + title (ingen description)
        # Normalisera vikter så de summerar till 1.0
        total = weight_type + weight_title + weight_desc
        w_type = (weight_type * 0.5) / total  # Dela type-vikt med google
        w_google = (weight_type * 0.5) / total
        w_title = (weight_title + weight_desc) / total  # Title tar över desc-vikt

        combined = (w_type * emb_types +
                   w_google * emb_google +
                   w_title * emb_titles)

        # Normalisera för cosine similarity
        combined = torch.nn.functional.normalize(combined, p=2, dim=1)

        return combined

    def _match_to_categories(self, product_embeddings: torch.Tensor, categories: list[dict],
                             cat_embeddings: torch.Tensor) -> list[tuple[dict, float]]:
        """Matcha produkter mot kategorier, returnera bästa match + confidence."""
        if len(categories) == 0:
            return [(None, 0.0)] * product_embeddings.shape[0]

        cat_emb = cat_embeddings.to(product_embeddings.device)
        similarities = torch.mm(product_embeddings, cat_emb.T)
        best_scores, best_indices = torch.max(similarities, dim=1)

        results = []
        for score, idx in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
            results.append((categories[idx], float(score)))

        return results

    def classify_batch(self, products: list[dict]) -> list[dict]:
        """Klassificera en batch hierarkiskt."""
        # Resultat för varje produkt
        results = [{
            'level_1': None, 'level_1_id': None, 'level_1_conf': 0.0,
            'level_2': None, 'level_2_id': None, 'level_2_conf': 0.0,
            'level_3': None, 'level_3_id': None, 'level_3_conf': 0.0,
            'level_4': None, 'level_4_id': None, 'level_4_conf': 0.0,
        } for _ in products]

        # === LEVEL 1 med gender-filtrering ===
        # Gruppera produkter efter gender
        gender_groups = {}
        for i, p in enumerate(products):
            gender = (p.get('gender', '') or '').lower()
            if gender not in gender_groups:
                gender_groups[gender] = []
            gender_groups[gender].append(i)

        # Skapa VIKTADE embeddings för Level 1 (använd config-vikter)
        product_emb_l1 = self.encode_products_weighted(
            products,
            weight_type=self.config.l1_weight_type,
            weight_title=self.config.l1_weight_title,
            weight_desc=self.config.l1_weight_desc
        )

        # Debug: visa produkters fält
        if products:
            p0 = products[0]
            logger.info(f"CSV kolumner: {list(p0.keys())}")
            logger.info(f"")
            logger.info(f"=== FÖRSTA 50 PRODUKTER ===")
            for i, p in enumerate(products[:50]):
                pt, t, d, gc = self._get_product_fields(p)
                logger.info(f"{i+1}. type='{pt}' | google='{gc}' | title='{t[:40]}...' | gender='{p.get('gender', '')}'")
            logger.info(f"")
            logger.info(f"Gender-grupper: {list(gender_groups.keys())}")

        # Matcha varje gender-grupp mot sina kategorier
        for gender, indices in gender_groups.items():
            level1_cats = self.cache.get_level(1, gender if gender else None)
            if not level1_cats:
                # Fallback till alla kategorier om gender inte hittas
                level1_cats = self.cache.get_level(1)
            if not level1_cats:
                continue

            # Berika Level 1 kategorier med barn-namn
            cache_key = f"level_1_gender_{gender}" if gender else "level_1"
            l1_emb, l1_cats = self.model.get_category_embeddings(level1_cats, cache_key, self.cache)

            # Hämta embeddings för denna gender-grupp
            group_emb = product_emb_l1[indices]
            l1_matches = self._match_to_categories(group_emb, l1_cats, l1_emb)

            for j, (cat, conf) in enumerate(l1_matches):
                idx = indices[j]
                if cat and conf >= self.config.min_confidence:
                    results[idx]['level_1'] = cat['name']
                    results[idx]['level_1_id'] = cat['id']
                    results[idx]['level_1_conf'] = round(conf, 4)

        # === LEVEL 2, 3, 4 ===
        # För djupare nivåer: använd VIKTADE embeddings (använd config-vikter)
        product_emb_full = self.encode_products_weighted(
            products,
            weight_type=self.config.l2_weight_type,
            weight_title=self.config.l2_weight_title,
            weight_desc=self.config.l2_weight_desc
        )

        for level in [2, 3, 4]:
            prev_level = level - 1
            prev_key = f'level_{prev_level}_id'

            # Gruppera produkter efter föräldra-kategori
            parent_groups = {}
            for i, result in enumerate(results):
                parent_id = result.get(prev_key)
                if parent_id:
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(i)

            # Matcha varje grupp mot sina barn-kategorier
            for parent_id, indices in parent_groups.items():
                children = self.cache.get_children(parent_id)
                if not children:
                    continue

                # Hämta embeddings för dessa barn (berikade med DERAS barn)
                cache_key = f"level_{level}_parent_{parent_id}"
                child_emb, child_cats = self.model.get_category_embeddings(children, cache_key, self.cache)

                # Hämta produkt-embeddings för denna grupp
                group_emb = product_emb_full[indices]

                # Matcha
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

        with tqdm(total=len(products), desc="Klassificerar hierarkiskt", unit="produkt") as pbar:
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
        # Lägg till hierarkiska resultat
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

    parser = argparse.ArgumentParser(description="Hierarkisk kategori-klassificering")
    parser.add_argument("--input", "-i", type=str, default=str(INPUT_DIR / "products.csv"))
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--batch-size", "-b", type=int, default=1024)
    parser.add_argument("--refresh-cache", action="store_true", help="Tvinga omladdning från DB")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--model", "-m", type=str, default="BAAI/bge-m3",
                       help="Embedding model (default: BAAI/bge-m3)")
    # Vikter för embedding (L1 = Level 1, L2 = Level 2+)
    parser.add_argument("--l1-weight-type", type=float, default=0.6, help="Vikt för product_type i Level 1")
    parser.add_argument("--l1-weight-title", type=float, default=0.4, help="Vikt för title i Level 1")
    parser.add_argument("--l2-weight-type", type=float, default=0.5, help="Vikt för product_type i Level 2+")
    parser.add_argument("--l2-weight-title", type=float, default=0.3, help="Vikt för title i Level 2+")
    parser.add_argument("--l2-weight-desc", type=float, default=0.2, help="Vikt för description i Level 2+")
    args = parser.parse_args()

    config = Config()
    config.model_name = args.model
    config.batch_size = args.batch_size
    config.min_confidence = args.min_confidence
    config.l1_weight_type = args.l1_weight_type
    config.l1_weight_title = args.l1_weight_title
    config.l1_weight_desc = 0.0  # Alltid 0 för Level 1
    config.l2_weight_type = args.l2_weight_type
    config.l2_weight_title = args.l2_weight_title
    config.l2_weight_desc = args.l2_weight_desc

    logger.info("=" * 60)
    logger.info("HIERARKISK KATEGORI-KLASSIFICERING")
    logger.info("=" * 60)
    logger.info(f"Modell: {config.model_name}")
    logger.info(f"Vikter L1: type={config.l1_weight_type}, title={config.l1_weight_title}, desc={config.l1_weight_desc}")
    logger.info(f"Vikter L2+: type={config.l2_weight_type}, title={config.l2_weight_title}, desc={config.l2_weight_desc}")

    # === Ladda kategorier (cache eller DB) ===
    category_cache = CategoryCache()

    if not args.refresh_cache and category_cache.load_from_cache():
        pass  # Laddad från cache
    else:
        if not config.get_db_url():
            logger.error("DATABASE_URL saknas!")
            sys.exit(1)
        await category_cache.load_from_db(config.get_db_url())

    # Visa kategori-struktur
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
    classifier = HierarchicalClassifier(config, category_cache, embedding_model)

    logger.info("=" * 60)
    logger.info(f"KLASSIFICERAR {len(products)} PRODUKTER")
    logger.info("=" * 60)

    start_time = time.time()
    results = classifier.classify_all(products)
    elapsed = time.time() - start_time

    # === Spara ===
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / f"{input_path.stem}_beta.csv"
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
