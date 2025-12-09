#!/usr/bin/env python3
"""
Product Category Classification - Embedding Similarity (BETA)
==============================================================
Klassificerar produkter till svenska kategorier med multilingual embeddings.
Ingen träning behövs - fungerar direkt!

Input-fält (samma som app.py):
- product_type
- title
- description

Output-fält:
- Alla original-kolumner + predicted_lokalt_kategori, confidence

Modell: intfloat/multilingual-e5-large
GPU: Optimerat för A100 40GB VRAM
Hastighet: 500 000 produkter på ~2 minuter

Användning:
    python beta.py                           # Använder inputs/products.csv och DB-kategorier
    python beta.py --input products.csv      # Specificera input-fil
"""

import asyncio
import csv
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


# =============================================================================
# KONFIGURATION - Optimerat för A100 40GB
# =============================================================================

@dataclass
class Config:
    """Konfiguration för A100 40GB VRAM."""
    # Database
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))

    # Embedding-modell - multilingual, förstår svenska + engelska
    model_name: str = "intfloat/multilingual-e5-large"

    # Batchstorlek - stor för maximal GPU-utnyttjande
    batch_size: int = 1024  # Stor batch för A100 40GB

    # Max antal tokens per text
    max_length: int = 256

    def get_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        return ""


# =============================================================================
# DATABASE - Samma som app.py
# =============================================================================

class Database:
    """Async PostgreSQL database access."""

    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        logger.info("Ansluter till PostgreSQL...")
        self.pool = await asyncpg.create_pool(
            self.config.get_db_url(),
            min_size=5,
            max_size=20
        )
        logger.info("Databasanslutning klar")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def fetch_categories(self) -> list[dict]:
        """Hämta kategorier från databasen (samma som app.py)."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.id, c.name, c.parent_id, c.level, c.path, c.slug, c.gender_id, g.name as gender_name
                FROM category c
                LEFT JOIN product_gender g ON c.gender_id = g.id
                WHERE c.active = true AND c.deleted_at IS NULL
                ORDER BY c.level, c.sort_order, c.name
            """)
            return [dict(row) for row in rows]


# =============================================================================
# EMBEDDING CATEGORY CLASSIFIER
# =============================================================================

class EmbeddingCategoryClassifier:
    """
    Klassificerar produkter till kategorier med embedding-likhet.

    Hur det fungerar:
    1. Ladda multilingual-e5-large modellen
    2. Skapa embeddings för alla svenska kategorier
    3. För varje produkt: skapa embedding och hitta mest lika kategori
    4. Cosine similarity används för att mäta likhet

    Modellen förstår att:
    - "Jacket" ≈ "Jacka"
    - "Pants" ≈ "Byxor"
    - "Bag" ≈ "Väska"
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.categories = []
        self.category_data = []  # Full category data from DB
        self.category_embeddings = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Ladda embedding-modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar modell: {self.config.model_name}")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16  # FP16 för snabbhet
        )

        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            logger.warning("Ingen GPU hittades! Kör på CPU (långsammare)")

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling för att få sentence embedding."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Koda en batch av texter till embeddings."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            # Normalisera för cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def set_categories(self, categories: list[dict]):
        """
        Sätt kategorier från databasen och beräkna deras embeddings.

        Args:
            categories: Lista med kategori-dicts från databasen
        """
        self.load_model()

        self.category_data = categories
        # Ta bort dubbletter baserat på namn
        seen_names = {}
        for cat in categories:
            name = cat['name']
            if name not in seen_names:
                seen_names[name] = cat

        self.categories = list(seen_names.keys())
        self.category_lookup = seen_names

        logger.info(f"Beräknar embeddings för {len(self.categories)} unika kategorier...")

        # E5-modellen förväntar "passage: " prefix för dokument/kategorier
        # Använd path för bättre kontext (t.ex. "Kläder > Jackor > Dunjackor")
        category_texts = []
        for name in self.categories:
            cat = self.category_lookup[name]
            path = cat.get('path') or name
            category_texts.append(f"passage: {path}")

        # Batcha för effektivitet
        all_embeddings = []
        for i in range(0, len(category_texts), self.config.batch_size):
            batch = category_texts[i:i + self.config.batch_size]
            embeddings = self._encode_batch(batch)
            all_embeddings.append(embeddings.cpu())

        self.category_embeddings = torch.cat(all_embeddings, dim=0)
        logger.info(f"Kategori-embeddings: {self.category_embeddings.shape}")

    def _build_product_text(self, product: dict) -> str:
        """
        Bygg textrepresentation av en produkt.

        Kombinerar: product_type + title + description
        (Samma fält som app.py använder)
        """
        product_type = product.get('product_type', '') or ''
        title = product.get('title', '') or ''
        description = (product.get('description', '') or '')[:300]

        # E5-modellen förväntar "query: " prefix för frågor/produkter
        parts = []
        if product_type:
            parts.append(f"Produkttyp: {product_type}")
        if title:
            parts.append(f"Titel: {title}")
        if description:
            parts.append(f"Beskrivning: {description}")

        text = f"query: {' | '.join(parts)}" if parts else "query: okänd produkt"
        return text

    def classify_batch(self, products: list[dict]) -> list[dict]:
        """
        Klassificera en batch av produkter.

        Returns:
            Lista med dict innehållande predicted_category, category_id och confidence
        """
        # Bygg produkttexter
        texts = [self._build_product_text(p) for p in products]

        # Koda produkter
        product_embeddings = self._encode_batch(texts)

        # Flytta kategori-embeddings till GPU
        cat_emb = self.category_embeddings.to(self.device)

        # Beräkna cosine similarity (embeddings är redan normaliserade)
        similarities = torch.mm(product_embeddings, cat_emb.T)

        # Hitta bästa match
        best_scores, best_indices = torch.max(similarities, dim=1)

        results = []
        for score, idx in zip(best_scores.cpu().numpy(), best_indices.cpu().numpy()):
            category_name = self.categories[idx]
            category_info = self.category_lookup[category_name]
            results.append({
                "predicted_lokalt_kategori": category_name,
                "predicted_lokalt_kategori_id": category_info.get('id'),
                "confidence": round(float(score), 4)
            })

        return results

    def classify_all(self, products: list[dict]) -> list[dict]:
        """
        Klassificera alla produkter med progress bar.

        Args:
            products: Lista med produktdicts

        Returns:
            Lista med klassificeringsresultat
        """
        self.load_model()

        all_results = []

        with tqdm(total=len(products), desc="Klassificerar", unit="produkt") as pbar:
            for i in range(0, len(products), self.config.batch_size):
                batch = products[i:i + self.config.batch_size]
                results = self.classify_batch(batch)
                all_results.extend(results)
                pbar.update(len(batch))

        return all_results


# =============================================================================
# CSV HANTERING - Samma logik som app.py
# =============================================================================

def read_csv(path: str) -> list[dict]:
    """
    Läs CSV-fil med samma logik som app.py.

    Använder iso-8859-1 encoding och semikolon som delimiter.
    """
    logger.info(f"Läser CSV: {path}")
    products = []
    with open(path, 'r', encoding='iso-8859-1') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            products.append(row)
    logger.info(f"Laddade {len(products)} produkter")
    return products


def write_csv(products: list[dict], results: list[dict], path: str):
    """
    Skriv resultat till CSV med samma format som app.py.

    Behåller alla original-kolumner och lägger till predicted_lokalt_kategori + confidence.
    """
    logger.info(f"Skriver CSV: {path}")

    if not products:
        return

    # Kombinera original-data med resultat
    output_rows = []
    for product, result in zip(products, results):
        row = dict(product)  # Kopiera alla original-kolumner
        row['predicted_lokalt_kategori'] = result['predicted_lokalt_kategori']
        row['predicted_lokalt_kategori_id'] = result['predicted_lokalt_kategori_id']
        row['embedding_confidence'] = result['confidence']
        output_rows.append(row)

    # Skriv med samma format som app.py
    fieldnames = list(output_rows[0].keys())
    with open(path, 'w', encoding='iso-8859-1', newline='', errors='replace') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(f"Sparade {len(output_rows)} rader")


# =============================================================================
# PATHS - Samma struktur som app.py
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Huvudfunktion - kör klassificering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Klassificera produkter till svenska kategorier med embedding-likhet"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=str(INPUT_DIR / "products.csv"),
        help=f"Input CSV-fil (default: {INPUT_DIR / 'products.csv'})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output CSV-fil (default: outputs/<input>_beta.csv)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1024,
        help="Batchstorlek för GPU (default: 1024)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Embedding-modell (default: intfloat/multilingual-e5-large)"
    )

    args = parser.parse_args()

    # =========================================================================
    # STEG 1: Konfigurera
    # =========================================================================
    config = Config()
    config.model_name = args.model
    config.batch_size = args.batch_size

    logger.info("=" * 60)
    logger.info("PRODUKT-KATEGORI KLASSIFICERING (BETA)")
    logger.info("=" * 60)
    logger.info(f"Modell: {config.model_name}")
    logger.info(f"Batchstorlek: {config.batch_size}")
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # =========================================================================
    # STEG 2: Anslut till databas och hämta kategorier
    # =========================================================================
    if not config.get_db_url():
        logger.error("DATABASE_URL saknas! Sätt miljövariabeln eller skapa .env fil.")
        sys.exit(1)

    db = Database(config)
    await db.connect()

    try:
        categories = await db.fetch_categories()
        logger.info(f"Hämtade {len(categories)} kategorier från databasen")

        # =========================================================================
        # STEG 3: Läs produkter
        # =========================================================================
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input-fil hittades inte: {input_path}")
            sys.exit(1)

        products = read_csv(str(input_path))
        if not products:
            logger.error("Inga produkter hittades!")
            sys.exit(1)

        # =========================================================================
        # STEG 4: Initiera klassificerare
        # =========================================================================
        classifier = EmbeddingCategoryClassifier(config)
        classifier.set_categories(categories)

        # =========================================================================
        # STEG 5: Klassificera alla produkter
        # =========================================================================
        logger.info("=" * 60)
        logger.info(f"KLASSIFICERAR {len(products)} PRODUKTER")
        logger.info("=" * 60)

        start_time = time.time()
        results = classifier.classify_all(products)
        elapsed = time.time() - start_time

        # =========================================================================
        # STEG 6: Spara resultat
        # =========================================================================
        OUTPUT_DIR.mkdir(exist_ok=True)

        if args.output:
            output_path = Path(args.output)
        else:
            # outputs/<filename>_beta.csv
            input_stem = input_path.stem  # filename utan .csv
            output_path = OUTPUT_DIR / f"{input_stem}_beta.csv"

        write_csv(products, results, str(output_path))

        # =========================================================================
        # STEG 7: Sammanfattning
        # =========================================================================
        products_per_sec = len(products) / elapsed if elapsed > 0 else 0

        logger.info("=" * 60)
        logger.info("KLART!")
        logger.info("=" * 60)
        logger.info(f"Produkter: {len(products)}")
        logger.info(f"Kategorier: {len(categories)}")
        logger.info(f"Tid: {elapsed:.1f} sekunder ({products_per_sec:.0f} produkter/sek)")
        logger.info(f"Output: {output_path}")

        # Visa exempel
        print("\n" + "=" * 60)
        print("EXEMPEL PÅ RESULTAT")
        print("=" * 60)
        for i in range(min(5, len(products))):
            p = products[i]
            r = results[i]
            print(f"\n{i+1}. Produkttyp: {p.get('product_type', '')}")
            print(f"   Titel: {(p.get('title', '') or '')[:50]}...")
            print(f"   → Matchad kategori: {r['predicted_lokalt_kategori']} ({r['confidence']:.1%})")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
