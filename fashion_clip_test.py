#!/usr/bin/env python3
"""
Test Multiple Fashion-Specific CLIP Models
==========================================
Kör samma klassificering med olika FASHION-specifika modeller för att jämföra.

Alla modeller är tränade på fashion/e-commerce data:

1. Marqo/marqo-fashionSigLIP     - SigLIP tränad på fashion (BEST)
2. Marqo/marqo-fashionCLIP       - CLIP tränad på fashion
3. patrickjohncyh/fashion-clip    - Original FashionCLIP
4. Marqo/marqo-ecommerce-embeddings-B  - E-commerce produkter
5. Marqo/marqo-ecommerce-embeddings-L  - E-commerce Large

Usage:
    python fashion_clip_test.py --model fashionsiglip
    python fashion_clip_test.py --model fashionclip
    python fashion_clip_test.py --model ecommerce-clip
    python fashion_clip_test.py --all  # Testa alla modeller
    python fashion_clip_test.py --all --limit 1000  # Snabb test med 1000 produkter
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "cache"


# =============================================================================
# FASHION-SPECIFIKA MODELLER
# =============================================================================

MODELS = {
    'fashionsiglip': {
        'name': 'Marqo/marqo-fashionSigLIP',
        'type': 'open_clip',
        'description': 'FashionSigLIP - Best fashion model (SigLIP-based)',
        'min_confidence': 0.15,
    },
    'fashionclip': {
        'name': 'Marqo/marqo-fashionCLIP',
        'type': 'open_clip',
        'description': 'FashionCLIP - Marqo version (ViT-B-16)',
        'min_confidence': 0.15,
    },
    'fashionclip-original': {
        'name': 'patrickjohncyh/fashion-clip',
        'type': 'transformers_clip',
        'description': 'Original FashionCLIP - First fashion CLIP model',
        'min_confidence': 0.15,
    },
    'fashion-mnist-clip': {
        'name': 'DunnBC22/clip-vit-base-patch16-fashion-mnist',
        'type': 'transformers_clip',
        'description': 'CLIP fine-tuned on Fashion-MNIST',
        'min_confidence': 0.15,
    },
    'ecommerce-clip': {
        'name': 'Marqo/marqo-ecommerce-embeddings-B',
        'type': 'open_clip',
        'description': 'Marqo E-commerce - Trained on product data',
        'min_confidence': 0.15,
    },
    'ecommerce-clip-L': {
        'name': 'Marqo/marqo-ecommerce-embeddings-L',
        'type': 'open_clip',
        'description': 'Marqo E-commerce Large - Bigger model',
        'min_confidence': 0.15,
    },
}


@dataclass
class Config:
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))
    model_key: str = 'fashionsiglip'
    batch_size: int = 256
    max_length: int = 77
    min_confidence: float = 0.15

    def get_db_url(self) -> str:
        return self.db_url or ""


# =============================================================================
# CATEGORY CACHE
# =============================================================================

class CategoryCache:
    GENDER_MAP = {'male': 1, 'female': 7, 'unisex': None}

    def __init__(self, cache_file: Path = CACHE_DIR / "categories.json"):
        self.cache_file = cache_file
        self.categories = []
        self.by_level = {}
        self.by_parent = {}
        self.by_id = {}
        self.by_level_gender = {}

    def load_from_cache(self) -> bool:
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
        self.cache_file.parent.mkdir(exist_ok=True)
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.categories, f, ensure_ascii=False, indent=2)

    async def load_from_db(self, db_url: str):
        logger.info("Hämtar kategorier från databas...")
        conn = await asyncpg.connect(db_url)
        try:
            rows = await conn.fetch("""
                SELECT id, name, parent_id, level, path, slug, gender_id
                FROM category WHERE active = true AND deleted_at IS NULL
                ORDER BY level, name
            """)
            self.categories = [dict(row) for row in rows]
            self._build_indexes()
            self.save_to_cache()
            logger.info(f"Hämtade {len(self.categories)} kategorier från DB")
        finally:
            await conn.close()

    def _build_indexes(self):
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
        if gender and gender.lower() in self.GENDER_MAP:
            gender_id = self.GENDER_MAP[gender.lower()]
            if gender_id is not None:
                return self.by_level_gender.get((level, gender_id), [])
        return self.by_level.get(level, [])

    def get_children(self, parent_id: int) -> list[dict]:
        return self.by_parent.get(parent_id, [])


# =============================================================================
# MULTI-MODEL LOADER
# =============================================================================

class MultiModelLoader:
    """Laddar olika CLIP/embedding modeller."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_info = MODELS[config.model_key]
        self.category_embeddings_cache = {}

    def load(self):
        if self.model is not None:
            return

        model_type = self.model_info['type']
        model_name = self.model_info['name']

        logger.info(f"Laddar modell: {model_name} (type: {model_type})")

        if model_type == 'open_clip':
            self._load_open_clip(model_name)
        elif model_type == 'transformers_clip':
            self._load_transformers_clip(model_name)
        elif model_type == 'transformers':
            self._load_transformers(model_name)
        else:
            raise ValueError(f"Okänd modelltyp: {model_type}")

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU: {gpu_name}")

    def _load_open_clip(self, model_name: str):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f'hf-hub:{model_name}'
        )
        self.tokenizer = open_clip.get_tokenizer(f'hf-hub:{model_name}')
        self.model.to(self.device)
        self.model.eval()
        self._encode_fn = self._encode_open_clip

    def _load_transformers_clip(self, model_name: str):
        from transformers import CLIPProcessor, CLIPModel
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch.float16)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self._encode_fn = self._encode_transformers_clip

    def _load_transformers(self, model_name: str):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16)
        self.model.to(self.device)
        self.model.eval()
        self._encode_fn = self._encode_transformers

    def _encode_open_clip(self, texts: list[str]) -> torch.Tensor:
        all_emb = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_emb.append(features)
        return torch.cat(all_emb, dim=0)

    def _encode_transformers_clip(self, texts: list[str]) -> torch.Tensor:
        all_emb = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                all_emb.append(features)
        return torch.cat(all_emb, dim=0)

    def _encode_transformers(self, texts: list[str]) -> torch.Tensor:
        all_emb = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                features = outputs.last_hidden_state.mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)
                all_emb.append(features)
        return torch.cat(all_emb, dim=0)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        self.load()
        return self._encode_fn(texts)

    def get_category_embeddings(self, categories: list[dict], cache_key: str,
                                   category_cache: CategoryCache = None) -> tuple[torch.Tensor, list[dict]]:
        if cache_key in self.category_embeddings_cache:
            return self.category_embeddings_cache[cache_key]

        if category_cache:
            names = []
            for cat in categories:
                children = category_cache.get_children(cat['id'])
                if children:
                    child_names = [c['name'] for c in children]
                    enriched = f"{cat['name']} clothing: {', '.join(child_names)}"
                    names.append(enriched)
                else:
                    names.append(f"{cat['name']} fashion category")
        else:
            names = [f"{cat['name']} fashion category" for cat in categories]

        embeddings = self.encode_text(names)
        self.category_embeddings_cache[cache_key] = (embeddings, categories)
        return embeddings, categories


# =============================================================================
# CLASSIFIER
# =============================================================================

class HierarchicalClassifier:
    def __init__(self, config: Config, category_cache: CategoryCache, model: MultiModelLoader):
        self.config = config
        self.cache = category_cache
        self.model = model

    def _get_product_fields(self, product: dict) -> tuple[str, str, str, str]:
        product_type = (product.get('product_type', '') or '').strip()
        title = (product.get('title', '') or '').strip()
        description = ((product.get('description', '') or '')[:200]).strip()
        google_cat = product.get('google_product_category', '') or ''
        if '>' in google_cat:
            parts = [p.strip() for p in google_cat.split('>')]
            google_cat = ' '.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        return product_type, title, description, google_cat

    def _build_product_text(self, product: dict) -> str:
        product_type, title, description, google_cat = self._get_product_fields(product)
        parts = []
        if product_type:
            parts.append(product_type)
        if google_cat:
            parts.append(google_cat)
        if title:
            clean_title = title.split(',')[0].strip()
            parts.append(clean_title)
        return ' '.join(parts) if parts else 'clothing item'

    def encode_products(self, products: list[dict]) -> torch.Tensor:
        texts = [self._build_product_text(p) for p in products]
        return self.model.encode_text(texts)

    def _match_to_categories(self, product_emb: torch.Tensor, categories: list[dict],
                             cat_emb: torch.Tensor) -> list[tuple[dict, float]]:
        if len(categories) == 0:
            return [(None, 0.0)] * product_emb.shape[0]
        cat_emb = cat_emb.to(product_emb.device)
        sims = product_emb @ cat_emb.T
        sims = (sims + 1) / 2  # Scale to 0-1
        best_scores, best_indices = torch.max(sims, dim=1)
        results = []
        for score, idx in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
            results.append((categories[idx], float(score)))
        return results

    def classify_batch(self, products: list[dict]) -> list[dict]:
        results = [{
            'level_1': None, 'level_1_id': None, 'level_1_conf': 0.0,
            'level_2': None, 'level_2_id': None, 'level_2_conf': 0.0,
            'level_3': None, 'level_3_id': None, 'level_3_conf': 0.0,
            'level_4': None, 'level_4_id': None, 'level_4_conf': 0.0,
        } for _ in products]

        gender_groups = {}
        for i, p in enumerate(products):
            gender = (p.get('gender', '') or '').lower()
            if gender not in gender_groups:
                gender_groups[gender] = []
            gender_groups[gender].append(i)

        product_emb = self.encode_products(products)

        for gender, indices in gender_groups.items():
            level1_cats = self.cache.get_level(1, gender if gender else None)
            if not level1_cats:
                level1_cats = self.cache.get_level(1)
            if not level1_cats:
                continue

            cache_key = f"level_1_gender_{gender}" if gender else "level_1"
            l1_emb, l1_cats = self.model.get_category_embeddings(level1_cats, cache_key, self.cache)
            group_emb = product_emb[indices]
            l1_matches = self._match_to_categories(group_emb, l1_cats, l1_emb)

            for j, (cat, conf) in enumerate(l1_matches):
                idx = indices[j]
                if cat and conf >= self.config.min_confidence:
                    results[idx]['level_1'] = cat['name']
                    results[idx]['level_1_id'] = cat['id']
                    results[idx]['level_1_conf'] = round(conf, 4)

        for level in [2, 3, 4]:
            prev_key = f'level_{level-1}_id'
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
                group_emb = product_emb[indices]
                matches = self._match_to_categories(group_emb, child_cats, child_emb)
                for j, (cat, conf) in enumerate(matches):
                    idx = indices[j]
                    if cat and conf >= self.config.min_confidence:
                        results[idx][f'level_{level}'] = cat['name']
                        results[idx][f'level_{level}_id'] = cat['id']
                        results[idx][f'level_{level}_conf'] = round(conf, 4)

        return results

    def classify_all(self, products: list[dict]) -> list[dict]:
        all_results = []
        model_name = MODELS[self.config.model_key]['name'].split('/')[-1]
        with tqdm(total=len(products), desc=f"Testing {model_name}", unit="prod") as pbar:
            for i in range(0, len(products), self.config.batch_size):
                batch = products[i:i + self.config.batch_size]
                results = self.classify_batch(batch)
                all_results.extend(results)
                pbar.update(len(batch))
        return all_results


# =============================================================================
# CSV
# =============================================================================

def read_csv(path: str) -> list[dict]:
    products = []
    with open(path, 'r', encoding='iso-8859-1') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            products.append(row)
    return products


def write_csv(products: list[dict], results: list[dict], path: str):
    if not products:
        return
    output_rows = []
    for product, result in zip(products, results):
        row = dict(product)
        for level in [1, 2, 3, 4]:
            row[f'level_{level}_kategori'] = result[f'level_{level}'] or ''
            row[f'level_{level}_kategori_id'] = result[f'level_{level}_id'] or ''
            row[f'level_{level}_confidence'] = result[f'level_{level}_conf']
        output_rows.append(row)
    fieldnames = list(output_rows[0].keys())
    with open(path, 'w', encoding='iso-8859-1', newline='', errors='replace') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)


# =============================================================================
# MAIN
# =============================================================================

async def test_model(model_key: str, products: list[dict], category_cache: CategoryCache,
                     input_stem: str) -> dict:
    """Testa en specifik modell."""
    config = Config()
    config.model_key = model_key
    config.min_confidence = MODELS[model_key]['min_confidence']

    model_info = MODELS[model_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"TESTAR: {model_info['name']}")
    logger.info(f"Beskrivning: {model_info['description']}")
    logger.info(f"{'='*60}")

    model = MultiModelLoader(config)
    classifier = HierarchicalClassifier(config, category_cache, model)

    start_time = time.time()
    results = classifier.classify_all(products)
    elapsed = time.time() - start_time

    # Spara resultat
    output_path = OUTPUT_DIR / f"{input_stem}_{model_key}.csv"
    write_csv(products, results, str(output_path))

    # Beräkna statistik
    l1_matches = sum(1 for r in results if r['level_1'])
    l2_matches = sum(1 for r in results if r['level_2'])
    avg_conf = sum(r['level_1_conf'] for r in results) / len(results) if results else 0

    stats = {
        'model': model_key,
        'name': model_info['name'],
        'time': elapsed,
        'products_per_sec': len(products) / elapsed,
        'l1_matches': l1_matches,
        'l1_match_rate': l1_matches / len(products) * 100,
        'l2_matches': l2_matches,
        'avg_confidence': avg_conf,
        'output': str(output_path),
    }

    logger.info(f"Resultat: {l1_matches}/{len(products)} L1 matches ({stats['l1_match_rate']:.1f}%)")
    logger.info(f"Tid: {elapsed:.1f}s ({stats['products_per_sec']:.0f} prod/s)")
    logger.info(f"Output: {output_path}")

    # Rensa GPU-minne
    del model
    del classifier
    torch.cuda.empty_cache()

    return stats


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Testa olika CLIP/embedding modeller")
    parser.add_argument("--input", "-i", type=str, default=str(INPUT_DIR / "products.csv"))
    parser.add_argument("--model", "-m", type=str, choices=list(MODELS.keys()),
                       help="Vilken modell att testa")
    parser.add_argument("--all", "-a", action="store_true", help="Testa alla modeller")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--limit", "-l", type=int, help="Begränsa antal produkter (för snabb test)")
    args = parser.parse_args()

    # Ladda kategorier
    category_cache = CategoryCache()
    if not args.refresh_cache and category_cache.load_from_cache():
        pass
    else:
        db_url = os.getenv('DATABASE_URL', '')
        if not db_url:
            logger.error("DATABASE_URL saknas!")
            sys.exit(1)
        await category_cache.load_from_db(db_url)

    # Ladda produkter
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input-fil hittades inte: {input_path}")
        sys.exit(1)

    products = read_csv(str(input_path))
    if args.limit:
        products = products[:args.limit]
    logger.info(f"Laddade {len(products)} produkter")

    OUTPUT_DIR.mkdir(exist_ok=True)
    input_stem = input_path.stem

    # Testa modeller
    all_stats = []

    if args.all:
        # Testa alla modeller
        for model_key in MODELS.keys():
            try:
                stats = await test_model(model_key, products, category_cache, input_stem)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Fel med {model_key}: {e}")
                all_stats.append({'model': model_key, 'error': str(e)})

        # Sammanfattning
        print("\n" + "=" * 80)
        print("SAMMANFATTNING - ALLA MODELLER")
        print("=" * 80)
        print(f"{'Modell':<20} {'L1 Match%':<12} {'Avg Conf':<12} {'Speed':<15} {'Output'}")
        print("-" * 80)
        for s in all_stats:
            if 'error' in s:
                print(f"{s['model']:<20} ERROR: {s['error']}")
            else:
                print(f"{s['model']:<20} {s['l1_match_rate']:.1f}%{'':<7} {s['avg_confidence']:.3f}{'':<7} {s['products_per_sec']:.0f} prod/s{'':<5} {Path(s['output']).name}")

    elif args.model:
        # Testa en specifik modell
        stats = await test_model(args.model, products, category_cache, input_stem)
        all_stats.append(stats)

    else:
        print("Ange --model <namn> eller --all för att testa modeller")
        print(f"\nTillgängliga modeller:")
        for key, info in MODELS.items():
            print(f"  {key:<20} - {info['description']}")


if __name__ == "__main__":
    asyncio.run(main())
