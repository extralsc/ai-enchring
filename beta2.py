#!/usr/bin/env python3
"""
Product Category Classification - RAG + LLM (BETA2)
====================================================
Klassificerar produkter till svenska kategorier med RAG-approach:

1. Embedding-sökning: Hitta TOP-K mest relevanta kategorier
2. LLM-beslut: Skicka endast dessa kandidater till LLM för slutgiltigt val

Detta ger:
- Snabbhet: LLM ser bara 5-10 kategorier istället för 800+
- Noggrannhet: LLM förstår semantik bättre än ren embedding-likhet
- Flerspråkighet: Embedding-modellen hanterar svenska/engelska automatiskt

Input-fält (samma som app.py):
- product_type
- title
- description

Modeller:
- Embedding: intfloat/multilingual-e5-large (flerspråkig)
- LLM: Qwen/Qwen2.5-7B-Instruct (via vLLM)

GPU: Optimerat för A100 40GB VRAM
Hastighet: ~500K produkter på 30-60 minuter (beroende på LLM)

Användning:
    python beta2.py                           # Använder inputs/products.csv
    python beta2.py --input products.csv      # Specificera input-fil
    python beta2.py --top-k 5                 # Antal kandidater till LLM
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
    embedding_model: str = "intfloat/multilingual-e5-large"

    # LLM för slutgiltigt beslut
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    use_vllm: bool = True  # Använd vLLM för snabbare inference

    # RAG-parametrar
    top_k: int = 5  # Antal kandidat-kategorier att skicka till LLM

    # Batchstorlek
    batch_size: int = 256  # Mindre batch pga LLM-anrop
    embedding_batch_size: int = 1024  # Större för ren embedding

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
# EMBEDDING MODEL - För RAG retrieval
# =============================================================================

class EmbeddingModel:
    """
    Multilingual embedding-modell för vektorsökning.

    Skapar embeddings för kategorier och produkter i samma semantiska rum.
    Fungerar med svenska och engelska blandat.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self):
        """Ladda embedding-modellen."""
        if self.model is not None:
            return

        logger.info(f"Laddar embedding-modell: {self.config.embedding_model}")

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.embedding_model)
        self.model = AutoModel.from_pretrained(
            self.config.embedding_model,
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

    def encode(self, texts: list[str], prefix: str = "query: ") -> torch.Tensor:
        """
        Koda texter till embeddings.

        Args:
            texts: Lista med texter
            prefix: "query: " för produkter, "passage: " för kategorier (E5-modellen)
        """
        self.load()

        # Lägg till prefix
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        all_embeddings = []
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(prefixed_texts), batch_size):
            batch = prefixed_texts[i:i + batch_size]

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
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)


# =============================================================================
# LLM MODEL - För slutgiltigt beslut
# =============================================================================

class LLMModel:
    """
    LLM för att välja rätt kategori från kandidater.

    Tar emot produkt + top-K kandidat-kategorier och väljer den bästa.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.use_vllm = False

    def load(self):
        """Ladda LLM-modellen."""
        if self.model is not None:
            return

        if self.config.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                logger.info(f"Laddar LLM med vLLM: {self.config.llm_model}")
                self.model = LLM(
                    model=self.config.llm_model,
                    dtype="half",
                    gpu_memory_utilization=0.4,  # Lämna rum för embedding-modellen
                    trust_remote_code=True,
                )
                self.sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=100,
                )
                self.use_vllm = True
                logger.info("vLLM laddad!")
                return
            except ImportError:
                logger.warning("vLLM ej installerat, använder transformers")
            except Exception as e:
                logger.warning(f"vLLM fel: {e}, använder transformers")

        # Fallback till transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Laddar LLM med transformers: {self.config.llm_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.use_vllm = False
        logger.info("LLM laddad!")

    def generate_batch(self, prompts: list[str]) -> list[str]:
        """Generera svar för flera prompts."""
        self.load()

        if self.use_vllm:
            outputs = self.model.generate(prompts, self.sampling_params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            # Transformers fallback
            results = []
            batch_size = 8

            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]

                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=2048
                    )
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    for j, response in enumerate(responses):
                        answer = response[len(batch_prompts[j]):].strip()
                        results.append(answer)

            return results


# =============================================================================
# RAG CATEGORY CLASSIFIER
# =============================================================================

class RAGCategoryClassifier:
    """
    RAG-baserad kategori-klassificerare.

    Steg 1: Skapa embeddings för alla kategorier (offline)
    Steg 2: För varje produkt:
            a) Skapa embedding
            b) Hitta top-K mest lika kategorier
            c) Skicka till LLM för slutgiltigt beslut
    """

    def __init__(self, config: Config):
        self.config = config
        self.embedding_model = EmbeddingModel(config)
        self.llm_model = LLMModel(config)
        self.categories = []
        self.category_embeddings = None
        self.category_lookup = {}

    def set_categories(self, categories: list[dict]):
        """
        Sätt kategorier och skapa embeddings.
        """
        self.categories = categories

        # Skapa lookup
        self.category_lookup = {cat['name']: cat for cat in categories}
        self.category_names = list(self.category_lookup.keys())

        logger.info(f"Skapar embeddings för {len(self.category_names)} kategorier...")

        # Använd path för bättre kontext
        category_texts = []
        for name in self.category_names:
            cat = self.category_lookup[name]
            path = cat.get('path') or name
            category_texts.append(path)

        # Skapa embeddings med "passage: " prefix
        self.category_embeddings = self.embedding_model.encode(category_texts, prefix="passage: ")
        logger.info(f"Kategori-embeddings: {self.category_embeddings.shape}")

    def _build_product_text(self, product: dict) -> str:
        """Bygg produkttext för embedding."""
        product_type = product.get('product_type', '') or ''
        title = product.get('title', '') or ''
        description = (product.get('description', '') or '')[:300]

        parts = []
        if product_type:
            parts.append(product_type)
        if title:
            parts.append(title)
        if description:
            parts.append(description)

        return ' '.join(parts) if parts else 'unknown product'

    def _find_top_k_categories(self, product_embeddings: torch.Tensor) -> list[list[tuple[str, float]]]:
        """
        Hitta top-K kandidat-kategorier för varje produkt.

        Returns:
            Lista av listor med (kategori_namn, likhet_score) tupler
        """
        cat_emb = self.category_embeddings.to(product_embeddings.device)

        # Cosine similarity
        similarities = torch.mm(product_embeddings, cat_emb.T)

        # Top-K för varje produkt
        top_scores, top_indices = torch.topk(similarities, k=self.config.top_k, dim=1)

        results = []
        for scores, indices in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
            candidates = []
            for score, idx in zip(scores, indices):
                candidates.append((self.category_names[idx], float(score)))
            results.append(candidates)

        return results

    def _build_llm_prompt(self, product: dict, candidates: list[tuple[str, float]]) -> str:
        """
        Bygg prompt för LLM med produkt + kandidat-kategorier.
        """
        product_type = product.get('product_type', '') or ''
        title = product.get('title', '') or ''
        description = (product.get('description', '') or '')[:200]

        # Bygg kandidatlista
        candidate_lines = []
        for i, (name, score) in enumerate(candidates, 1):
            candidate_lines.append(f"{i}. {name} (likhet: {score:.2f})")

        prompt = f"""Du är en svensk e-handels-expert. Välj den bästa kategorin för produkten.

Produkt:
- Typ: {product_type}
- Titel: {title}
- Beskrivning: {description}

Kandidat-kategorier (sorterade efter likhet):
{chr(10).join(candidate_lines)}

Uppgift: Välj den BÄSTA kategorin från listan ovan. Produkttexten kan vara på svenska eller engelska - välj baserat på semantik, inte språk.

Svara ENDAST med:
KATEGORI: [exakt kategorinamn från listan]
KONFIDENS: [hög/medel/låg]"""

        return prompt

    def _parse_llm_response(self, response: str, candidates: list[tuple[str, float]]) -> tuple[str, float]:
        """
        Parsa LLM-svar och returnera vald kategori + konfidens.
        """
        lines = response.strip().split('\n')

        category_name = None
        confidence = 0.7

        for line in lines:
            line = line.strip()
            line_lower = line.lower()

            if line_lower.startswith('kategori'):
                if ':' in line:
                    category_name = line.split(':', 1)[1].strip()
            elif line_lower.startswith('konfidens'):
                if ':' in line:
                    conf_str = line.split(':', 1)[1].strip().lower()
                    if 'hög' in conf_str or 'high' in conf_str:
                        confidence = 0.95
                    elif 'medel' in conf_str or 'medium' in conf_str:
                        confidence = 0.75
                    else:
                        confidence = 0.5

        # Validera att kategorin finns i kandidatlistan
        candidate_names = [c[0] for c in candidates]
        if category_name and category_name in candidate_names:
            return category_name, confidence

        # Fallback: ta första kandidaten
        if candidates:
            return candidates[0][0], candidates[0][1]

        return "unknown", 0.0

    def classify_batch(self, products: list[dict]) -> list[dict]:
        """
        Klassificera en batch av produkter med RAG + LLM.
        """
        # Steg 1: Skapa embeddings för produkterna
        product_texts = [self._build_product_text(p) for p in products]
        product_embeddings = self.embedding_model.encode(product_texts, prefix="query: ")

        if torch.cuda.is_available():
            product_embeddings = product_embeddings.to("cuda")

        # Steg 2: Hitta top-K kandidater
        all_candidates = self._find_top_k_categories(product_embeddings)

        # Steg 3: Bygg LLM-prompts
        prompts = []
        for product, candidates in zip(products, all_candidates):
            prompt = self._build_llm_prompt(product, candidates)
            prompts.append(prompt)

        # Steg 4: LLM-beslut
        responses = self.llm_model.generate_batch(prompts)

        # Steg 5: Parsa svar
        results = []
        for response, candidates in zip(responses, all_candidates):
            category_name, confidence = self._parse_llm_response(response, candidates)

            category_info = self.category_lookup.get(category_name, {})
            results.append({
                "predicted_lokalt_kategori": category_name,
                "predicted_lokalt_kategori_id": category_info.get('id'),
                "confidence": round(confidence, 4),
                "top_candidates": "; ".join([f"{c[0]} ({c[1]:.2f})" for c in candidates[:3]])
            })

        return results

    def classify_all(self, products: list[dict]) -> list[dict]:
        """
        Klassificera alla produkter med progress bar.
        """
        all_results = []

        with tqdm(total=len(products), desc="RAG+LLM Klassificering", unit="produkt") as pbar:
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
    """Läs CSV-fil med samma logik som app.py."""
    logger.info(f"Läser CSV: {path}")
    products = []
    with open(path, 'r', encoding='iso-8859-1') as f:
        reader = csv.DictReader(f, delimiter=';', quotechar='"')
        for row in reader:
            products.append(row)
    logger.info(f"Laddade {len(products)} produkter")
    return products


def write_csv(products: list[dict], results: list[dict], path: str):
    """Skriv resultat till CSV med samma format som app.py."""
    logger.info(f"Skriver CSV: {path}")

    if not products:
        return

    output_rows = []
    for product, result in zip(products, results):
        row = dict(product)
        row['predicted_lokalt_kategori'] = result['predicted_lokalt_kategori']
        row['predicted_lokalt_kategori_id'] = result['predicted_lokalt_kategori_id']
        row['rag_confidence'] = result['confidence']
        row['top_candidates'] = result['top_candidates']
        output_rows.append(row)

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
    """Huvudfunktion - kör RAG+LLM klassificering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Klassificera produkter med RAG (embedding) + LLM"
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
        help="Output CSV-fil (default: outputs/<input>_beta2.csv)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=256,
        help="Batchstorlek (default: 256)"
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="Antal kandidat-kategorier till LLM (default: 5)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="intfloat/multilingual-e5-large",
        help="Embedding-modell (default: intfloat/multilingual-e5-large)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="LLM-modell (default: Qwen/Qwen2.5-7B-Instruct)"
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Använd transformers istället för vLLM"
    )

    args = parser.parse_args()

    # =========================================================================
    # STEG 1: Konfigurera
    # =========================================================================
    config = Config()
    config.embedding_model = args.embedding_model
    config.llm_model = args.llm_model
    config.batch_size = args.batch_size
    config.top_k = args.top_k
    config.use_vllm = not args.no_vllm

    logger.info("=" * 60)
    logger.info("PRODUKT-KATEGORI KLASSIFICERING (BETA2 - RAG+LLM)")
    logger.info("=" * 60)
    logger.info(f"Embedding-modell: {config.embedding_model}")
    logger.info(f"LLM-modell: {config.llm_model}")
    logger.info(f"Top-K kandidater: {config.top_k}")
    logger.info(f"Batchstorlek: {config.batch_size}")
    logger.info(f"vLLM: {'Ja' if config.use_vllm else 'Nej'}")
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
        # STEG 4: Initiera RAG-klassificerare
        # =========================================================================
        classifier = RAGCategoryClassifier(config)
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
            input_stem = input_path.stem
            output_path = OUTPUT_DIR / f"{input_stem}_beta2.csv"

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
        logger.info(f"Tid: {elapsed:.1f} sekunder ({elapsed/60:.1f} minuter)")
        logger.info(f"Hastighet: {products_per_sec:.0f} produkter/sek")
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
            print(f"   Top kandidater: {r['top_candidates']}")
            print(f"   → Vald kategori: {r['predicted_lokalt_kategori']} ({r['confidence']:.0%})")

    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
