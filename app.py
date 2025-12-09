#!/usr/bin/env python3
"""
Product Data Enrichment Pipeline - OPTIMIZED FOR SPEED
Maximizes GPU/CPU utilization on A10 (24GB VRAM, 30 vCPUs, 226GB RAM)
Target: 40K products in ~5 minutes
"""

import asyncio
import csv
import logging
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Optional

import aiohttp
import asyncpg
import numpy as np
import torch
# Note: Dataset not needed - pipeline handles lists directly
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import open_clip
from transformers import pipeline
from sentence_transformers import SentenceTransformer
# GPT-SW3 translator uses AutoModelForCausalLM (loaded in NeuralTranslator class)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION - OPTIMIZED FOR A10
# =============================================================================

@dataclass
class Config:
    """Pipeline configuration - MAXIMIZED for A10 GPU (24GB VRAM, 30 vCPUs, 226GB RAM)."""
    # Database
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))

    # Processing - OPTIMIZED for A10 (23GB VRAM)
    batch_size: int = 512  # Image processing batch (reduced to avoid OOM)
    max_concurrent_downloads: int = 1000  # Max async downloads
    download_timeout: int = 10  # Faster timeout
    use_fp16: bool = True
    num_workers: int = 16  # CPU workers for image preprocessing

    # Classification batch sizes
    bart_batch_size: int = 128  # Zero-shot classification batch

    # Speed vs accuracy tradeoff
    # Zero-shot category is SLOW (~80 labels) but accurate
    # Sentence similarity is FAST but less accurate
    # Set to False for ~3x faster processing
    use_zeroshot_category: bool = False  # Disable slow zero-shot, use sentence similarity

    # Neural translation - translates descriptions to English for better understanding
    # Handles 70+ suppliers with different languages
    use_neural_translation: bool = True  # Enable for multi-language support

    # LLM-based category matching - uses a local LLM for accurate understanding
    # Much more accurate but slower than embedding matching
    use_llm_category: bool = True  # Enable for best accuracy
    llm_model: str = "Qwen/Qwen2.5-3B-Instruct"  # Good multilingual LLM
    use_vllm: bool = True  # Use vLLM for faster, stable inference
    llm_confidence_threshold: float = 0.90  # Only use LLM if embedding confidence < this (saves time)

    # Models - OPTIMIZED for A10's 24GB VRAM
    # FashionSigLIP outperforms FashionCLIP on fashion benchmarks
    fashion_model: str = "Marqo/marqo-fashionSigLIP"
    # Larger multilingual model (~1GB) - better accuracy for Swedish categories
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # DeBERTa-v3-base-zeroshot-v2.0 is the newest/best zero-shot classifier (2024)
    zero_shot_model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"

    # Prompts for CLIP
    color_prompts: list = field(default_factory=lambda: [
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "black", "white", "gray", "grey", "brown", "beige", "navy",
        "burgundy", "teal", "turquoise", "coral", "maroon", "olive",
        "cream", "gold", "silver", "khaki", "tan", "charcoal",
        "multicolor", "patterned", "striped", "plaid"
    ])

    category_prompts: list = field(default_factory=lambda: [
        "jacket", "coat", "blazer", "vest", "cardigan", "sweater", "hoodie",
        "t-shirt", "shirt", "blouse", "polo", "tank top", "crop top",
        "jeans", "pants", "trousers", "shorts", "skirt", "dress",
        "suit", "jumpsuit", "romper", "overalls",
        "underwear", "bra", "panties", "boxers", "briefs", "socks",
        "shoes", "sneakers", "boots", "sandals", "heels", "flats", "loafers",
        "bag", "backpack", "purse", "wallet", "belt", "scarf", "hat", "gloves",
        "swimwear", "bikini", "activewear", "leggings", "sports bra",
        "pajamas", "robe", "lingerie"
    ])

    size_type_labels: list = field(default_factory=lambda: [
        "jeans/pants/trousers",
        "t-shirt/shirt/top/blouse",
        "sweater/fleece/hoodie/crew",
        "jacket/coat/outerwear",
        "dress",
        "skirt",
        "shorts",
        "shoes/boots/sneakers",
        "socks",
        "underwear/lingerie",
        "swimwear/bikini",
        "accessories/belt/hat/scarf",
        "leggings/tights",
        "one-size"
    ])

    # Minimum confidence to consider a mapping valid (otherwise return "unknown")
    min_mapping_confidence: float = 0.5

    # Common category normalization rules for suggestions
    category_normalization: dict = field(default_factory=lambda: {
        # Detected → Normalized suggestion (for when no local match)
        'leggings': 'Athletic Leggings',
        'tights': 'Athletic Tights',
        'activewear': 'Activewear',
        'sweater': 'Sweater',
        'hoodie': 'Hoodie',
        't-shirt': 'T-Shirt',
        'jacket': 'Jacket',
        'pants': 'Pants',
        'jeans': 'Jeans',
        'shorts': 'Shorts',
        'dress': 'Dress',
        'skirt': 'Skirt',
        'coat': 'Coat',
        'blazer': 'Blazer',
        'cardigan': 'Cardigan',
        'vest': 'Vest',
        'shirt': 'Shirt',
        'blouse': 'Blouse',
        'polo': 'Polo Shirt',
        'tank top': 'Tank Top',
        'crop top': 'Crop Top',
        'jumpsuit': 'Jumpsuit',
        'romper': 'Romper',
        'overalls': 'Overalls',
        'underwear': 'Underwear',
        'bra': 'Bra',
        'socks': 'Socks',
        'shoes': 'Shoes',
        'sneakers': 'Sneakers',
        'boots': 'Boots',
        'sandals': 'Sandals',
        'heels': 'Heels',
        'flats': 'Flats',
        'loafers': 'Loafers',
        'bag': 'Bag',
        'backpack': 'Backpack',
        'purse': 'Purse',
        'wallet': 'Wallet',
        'belt': 'Belt',
        'scarf': 'Scarf',
        'hat': 'Hat',
        'gloves': 'Gloves',
        'swimwear': 'Swimwear',
        'bikini': 'Bikini',
        'sports bra': 'Sports Bra',
        'pajamas': 'Pajamas',
        'robe': 'Robe',
        'lingerie': 'Lingerie',
    })

    def get_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        return ""


# =============================================================================
# NEURAL MACHINE TRANSLATION - Using AI-Sweden GPT-SW3 Translator
# =============================================================================

class NeuralTranslator:
    """Neural machine translation using AI-Sweden GPT-SW3-6.7B-v2-translator.

    High-quality Swedish↔English translation optimized for Nordic languages.
    Uses vLLM for fast, efficient inference on GPU.
    """

    MODEL_NAME = "AI-Sweden-Models/gpt-sw3-6.7b-v2-translator"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.use_vllm = False

    def _load_model(self):
        """Lazy load GPT-SW3 translator model."""
        if self.model is not None:
            return

        # Try vLLM first (faster, more memory efficient)
        try:
            from vllm import LLM, SamplingParams
            logger.info(f"Loading GPT-SW3 translator with vLLM: {self.MODEL_NAME}...")
            self.model = LLM(
                model=self.MODEL_NAME,
                dtype="half",
                gpu_memory_utilization=0.25,  # Conservative - leave room for other models
                trust_remote_code=True,
                max_model_len=512,  # Translation doesn't need long context
            )
            self.sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=256,
                stop=["<s>", "</s>", "<|endoftext|>"],
            )
            self.use_vllm = True
            logger.info("GPT-SW3 translator loaded with vLLM!")
            return
        except ImportError:
            logger.warning("vLLM not installed, falling back to transformers")
        except Exception as e:
            logger.warning(f"vLLM failed for GPT-SW3: {e}, falling back to transformers")

        # Fallback to transformers pipeline
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading GPT-SW3 translator with transformers: {self.MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.use_vllm = False
        logger.info("GPT-SW3 translator loaded with transformers!")

    def _build_prompt(self, text: str, direction: str = "sv_to_en") -> str:
        """Build translation prompt for GPT-SW3."""
        if direction == "sv_to_en":
            return f"<|endoftext|><s>User: Översätt till Engelska från Svenska\n{text}<s>Bot:"
        else:  # en_to_sv
            return f"<|endoftext|><s>User: Översätt till Svenska från Engelska\n{text}<s>Bot:"

    def translate_batch(self, texts: list[str], src_lang: str = 'sv') -> list[str]:
        """Translate batch of texts. Supports Swedish↔English.

        Args:
            texts: List of texts to translate
            src_lang: Source language ('sv' for Swedish→English, 'en' for English→Swedish)
        """
        if not texts:
            return []

        # GPT-SW3 is optimized for Swedish↔English
        # For other languages, return original (they'll use dictionary fallback)
        if src_lang not in ('sv', 'en'):
            logger.debug(f"GPT-SW3 doesn't support {src_lang}, returning original texts")
            return texts

        self._load_model()

        direction = "sv_to_en" if src_lang == 'sv' else "en_to_sv"

        # Truncate long texts
        texts = [t[:400] if t else "" for t in texts]

        # Build prompts
        prompts = [self._build_prompt(t, direction) for t in texts]

        if self.use_vllm:
            # vLLM - process all at once
            outputs = self.model.generate(prompts, self.sampling_params)
            translations = []
            for output in outputs:
                translation = output.outputs[0].text.strip()
                # Clean up any trailing special tokens
                for stop in ["<s>", "</s>", "<|endoftext|>"]:
                    if translation.endswith(stop):
                        translation = translation[:-len(stop)].strip()
                translations.append(translation)
            return translations
        else:
            # Transformers fallback - batch processing
            translations = []
            batch_size = 8

            with torch.no_grad():
                for i in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[i:i + batch_size]

                    inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                    responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                    for j, response in enumerate(responses):
                        # Extract translation (after the prompt)
                        prompt_len = len(batch_prompts[j])
                        translation = response[prompt_len:].strip()
                        # Clean up special tokens
                        for stop in ["<s>", "</s>", "<|endoftext|>", "Bot:"]:
                            translation = translation.replace(stop, "").strip()
                        translations.append(translation)

            return translations

    def detect_and_translate(self, text: str) -> tuple[str, str]:
        """Detect language and translate to English if needed.

        Returns: (translated_text, detected_language)
        """
        if not text:
            return "", "unknown"

        # Simple language detection based on common words
        text_lower = text.lower()

        # Swedish indicators
        swedish_words = ['och', 'för', 'med', 'som', 'är', 'att', 'av', 'på', 'den', 'det', 'i', 'till']
        # German/French etc. - GPT-SW3 focuses on Swedish, so we skip others
        german_words = ['und', 'für', 'mit', 'ist', 'das', 'die', 'der', 'ein', 'eine', 'zu']
        french_words = ['et', 'pour', 'avec', 'est', 'le', 'la', 'les', 'un', 'une', 'de']

        words = text_lower.split()

        swedish_count = sum(1 for w in words if w in swedish_words)
        german_count = sum(1 for w in words if w in german_words)
        french_count = sum(1 for w in words if w in french_words)

        if swedish_count > german_count and swedish_count > french_count and swedish_count > 1:
            # Swedish text - translate to English
            translated = self.translate_batch([text], 'sv')[0]
            return translated, 'sv'
        elif german_count > 1 or french_count > 1:
            # Other languages - GPT-SW3 doesn't support, return original
            return text, 'other'
        else:
            # Assume English - no translation needed
            return text, 'en'


# Global translator instance (lazy loaded)
_translator = None

def get_translator() -> NeuralTranslator:
    """Get or create the global translator instance."""
    global _translator
    if _translator is None:
        _translator = NeuralTranslator()
    return _translator


# =============================================================================
# BILINGUAL DICTIONARY - WikiDict Swedish↔English
# =============================================================================

class BilingualDictionary:
    """Fast Swedish↔English dictionary using WikiDict SQLite databases.

    Provides word-level translations for better cross-language category matching.
    ~70K entries in each direction, covers most fashion terminology.
    """

    def __init__(self, dict_dir: str = "dictionaries"):
        self.dict_dir = Path(dict_dir)
        self.sv_en = {}  # Swedish → English
        self.en_sv = {}  # English → Swedish
        self._loaded = False

    def load(self):
        """Load dictionaries into memory for fast lookup."""
        if self._loaded:
            return

        sv_en_path = self.dict_dir / "sv-en.sqlite3"
        en_sv_path = self.dict_dir / "en-sv.sqlite3"

        if sv_en_path.exists():
            conn = sqlite3.connect(str(sv_en_path))
            cursor = conn.cursor()
            cursor.execute("SELECT written_rep, trans_list FROM simple_translation")
            for word, translations in cursor.fetchall():
                # Store first translation (most common) + all alternatives
                self.sv_en[word.lower()] = translations.split(" | ")[0] if translations else ""
            conn.close()
            logger.info(f"Loaded {len(self.sv_en)} Swedish→English translations")
        else:
            logger.warning(f"Swedish→English dictionary not found: {sv_en_path}")

        if en_sv_path.exists():
            conn = sqlite3.connect(str(en_sv_path))
            cursor = conn.cursor()
            cursor.execute("SELECT written_rep, trans_list FROM simple_translation")
            for word, translations in cursor.fetchall():
                self.en_sv[word.lower()] = translations.split(" | ")[0] if translations else ""
            conn.close()
            logger.info(f"Loaded {len(self.en_sv)} English→Swedish translations")
        else:
            logger.warning(f"English→Swedish dictionary not found: {en_sv_path}")

        self._loaded = True

    def translate_sv_to_en(self, word: str) -> str:
        """Translate Swedish word to English. Returns empty string if not found."""
        return self.sv_en.get(word.lower().strip(), "")

    def translate_en_to_sv(self, word: str) -> str:
        """Translate English word to Swedish. Returns empty string if not found."""
        return self.en_sv.get(word.lower().strip(), "")

    def enrich_text(self, text: str) -> str:
        """Enrich text with translations in both directions.

        For each word, if a translation exists, append it.
        This helps the multilingual model match across languages.
        """
        if not self._loaded:
            self.load()

        words = text.lower().split()
        enriched_words = list(words)  # Keep original words

        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip(".,!?()[]{}\"'")

            # Try Swedish → English
            en_trans = self.sv_en.get(clean_word, "")
            if en_trans:
                enriched_words.append(en_trans)

            # Try English → Swedish
            sv_trans = self.en_sv.get(clean_word, "")
            if sv_trans:
                enriched_words.append(sv_trans)

        return " ".join(enriched_words)


# Global dictionary instance (lazy loaded)
_dictionary = None

def get_dictionary() -> BilingualDictionary:
    """Get or create the global dictionary instance."""
    global _dictionary
    if _dictionary is None:
        _dictionary = BilingualDictionary()
    return _dictionary


# =============================================================================
# DATABASE
# =============================================================================

class Database:
    """Async PostgreSQL database access."""

    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        logger.info("Connecting to PostgreSQL...")
        self.pool = await asyncpg.create_pool(
            self.config.get_db_url(),
            min_size=10,
            max_size=50  # More connections for 30 vCPUs
        )
        logger.info("Database connected")

    async def close(self):
        if self.pool:
            await self.pool.close()

    async def fetch_colors(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name, hex_code FROM color
                WHERE active = true AND deleted_at IS NULL
                ORDER BY sort_order, name
            """)
            return [{"id": row["id"], "name": row["name"], "hex_code": row["hex_code"]} for row in rows]

    async def fetch_categories(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT c.id, c.name, c.parent_id, c.level, c.path, c.slug, c.gender_id, g.name as gender_name
                FROM category c
                LEFT JOIN product_gender g ON c.gender_id = g.id
                WHERE c.active = true AND c.deleted_at IS NULL
                ORDER BY c.level, c.sort_order, c.name
            """)
            return [dict(row) for row in rows]

    async def fetch_genders(self) -> list[dict]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, name FROM product_gender
                WHERE active = true AND deleted_at IS NULL
            """)
            return [{"id": row["id"], "name": row["name"]} for row in rows]


# =============================================================================
# FAST IMAGE DOWNLOADER - Streams directly to memory
# =============================================================================

class FastImageDownloader:
    """Ultra-fast async image downloader with high concurrency."""

    def __init__(self, config: Config):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_downloads)
        self.connector = aiohttp.TCPConnector(
            limit=config.max_concurrent_downloads,
            limit_per_host=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        self.timeout = aiohttp.ClientTimeout(total=config.download_timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {"success": 0, "failed": 0}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def download_one(self, url: str) -> Optional[Image.Image]:
        """Download single image directly to PIL Image (no disk)."""
        if not url or not url.startswith('http'):
            return None

        async with self.semaphore:
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        data = await response.read()
                        image = Image.open(BytesIO(data)).convert('RGB')
                        self.stats["success"] += 1
                        return image
            except:
                pass
            self.stats["failed"] += 1
            return None

    async def download_batch(self, urls: list[str]) -> list[Optional[Image.Image]]:
        """Download batch of images concurrently."""
        tasks = [self.download_one(url) for url in urls]
        return await asyncio.gather(*tasks)


# =============================================================================
# FASHION IMAGE PROCESSOR - Using Marqo FashionSigLIP
# =============================================================================

class FashionImageProcessor:
    """Fashion-specific image processor using Marqo FashionSigLIP model.

    +57% better recall than generic CLIP for fashion/clothing detection.
    Trained specifically on fashion colors, categories, materials, and styles.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"FashionSigLIP using device: {self.device}")

        # Load fashion-specific model using open_clip
        logger.info(f"Loading {config.fashion_model}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            f'hf-hub:{config.fashion_model}'
        )
        self.tokenizer = open_clip.get_tokenizer(f'hf-hub:{config.fashion_model}')

        # Optimize for speed
        self.model = self.model.to(self.device)
        if config.use_fp16 and self.device == "cuda":
            self.model = self.model.half()
        self.model.eval()

        # Compile model for 20-40% speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("FashionSigLIP compiled with torch.compile() for faster inference")
            except Exception as e:
                logger.warning(f"torch.compile() failed for FashionSigLIP: {e}")

        # Pre-compute text embeddings for colors and categories
        self._precompute_embeddings()

    def set_local_colors(self, local_colors: list[dict]):
        """Set local colors from database - CLIP will detect these exact colors."""
        self.local_color_names = [c["name"] for c in local_colors]
        self.local_color_ids = [c["id"] for c in local_colors]

        # Create embeddings for LOCAL colors
        color_texts = [f"{c} colored clothing" for c in self.local_color_names]
        with torch.no_grad():
            color_tokens = self.tokenizer(color_texts).to(self.device)
            self.color_embeddings = self.model.encode_text(color_tokens, normalize=True)
            if self.config.use_fp16:
                self.color_embeddings = self.color_embeddings.half()

        logger.info(f"FashionSigLIP: Using {len(self.local_color_names)} LOCAL colors: {self.local_color_names[:10]}...")

    @torch.no_grad()
    def _precompute_embeddings(self):
        """Pre-compute category embeddings (colors are set from database later)."""
        # Category prompts - fashion-specific phrasing
        category_texts = [f"a {c}" for c in self.config.category_prompts]
        category_tokens = self.tokenizer(category_texts).to(self.device)
        self.category_embeddings = self.model.encode_text(category_tokens, normalize=True)
        if self.config.use_fp16:
            self.category_embeddings = self.category_embeddings.half()

        # Initialize with config colors as fallback (will be replaced by local colors)
        self.local_color_names = self.config.color_prompts
        self.local_color_ids = [None] * len(self.config.color_prompts)
        color_texts = [f"{c} colored clothing" for c in self.local_color_names]
        color_tokens = self.tokenizer(color_texts).to(self.device)
        self.color_embeddings = self.model.encode_text(color_tokens, normalize=True)
        if self.config.use_fp16:
            self.color_embeddings = self.color_embeddings.half()

        logger.info(f"Pre-computed embeddings for {len(self.config.category_prompts)} categories")

    def _preprocess_worker(self, img: Image.Image) -> torch.Tensor:
        """Preprocess single image (for parallel execution)."""
        return self.preprocess(img)

    @torch.no_grad()
    def process_batch(self, images: list[Image.Image]) -> list[dict]:
        """Process batch of images - returns color and category predictions."""
        if not images:
            return []

        # Filter valid images
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [images[i] for i in valid_indices]

        if not valid_images:
            return [{"color": None, "color_id": None, "category": None, "color_conf": 0, "category_conf": 0} for _ in images]

        # PARALLEL preprocessing using ThreadPoolExecutor (CPU-bound)
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            preprocessed = list(executor.map(self._preprocess_worker, valid_images))

        processed_images = torch.stack(preprocessed).to(self.device)
        if self.config.use_fp16:
            processed_images = processed_images.half()

        # Get image features (normalized)
        image_features = self.model.encode_image(processed_images, normalize=True)

        # Compute similarities (scale by 100 as per FashionSigLIP convention)
        color_sims = (100.0 * image_features @ self.color_embeddings.T).softmax(dim=-1)
        category_sims = (100.0 * image_features @ self.category_embeddings.T).softmax(dim=-1)

        color_probs, color_idx = color_sims.max(dim=-1)
        category_probs, category_idx = category_sims.max(dim=-1)

        # Build results - use LOCAL colors from database
        results = [{"color": None, "color_id": None, "category": None, "color_conf": 0.0, "category_conf": 0.0} for _ in images]
        for i, valid_idx in enumerate(valid_indices):
            cidx = color_idx[i].item()
            results[valid_idx] = {
                "color": self.local_color_names[cidx],
                "color_id": self.local_color_ids[cidx],
                "category": self.config.category_prompts[category_idx[i].item()],
                "color_conf": round(color_probs[i].item(), 3),
                "category_conf": round(category_probs[i].item(), 3)
            }
        return results


# =============================================================================
# TAXONOMY MAPPER - Uses sentence embeddings
# =============================================================================

class TaxonomyMapper:
    """Maps values to local taxonomy using sentence embeddings.

    Swedish-only matching with gender filtering.
    Categories are filtered by product gender before matching.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.sentence_model)
        # Move to GPU for faster encoding
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        self.local_colors = []
        self.local_categories = []
        self.local_genders = []

        # Categories grouped by gender_id for fast filtering
        self.categories_by_gender = {}  # gender_id -> list of category indices
        self.category_embeddings_by_gender = {}  # gender_id -> embeddings tensor

        # Load WikiDict dictionary for word-level translations (fast, for enrichment)
        self.dictionary = get_dictionary()
        self.dictionary.load()

        # Load GPT-SW3 translator for high-quality Swedish↔English translation
        self.translator = get_translator()

    def set_local_colors(self, colors: list[dict]):
        self.local_colors = colors
        names = [c["name"] for c in colors]
        self.color_embeddings = self.model.encode(names, convert_to_tensor=True, show_progress_bar=False)

    def set_local_categories(self, categories: list[dict]):
        """Set categories and group by gender for filtered matching."""
        self.local_categories = categories

        # Group categories by gender_id
        self.categories_by_gender = {}
        for i, c in enumerate(categories):
            gender_id = c.get("gender_id")
            if gender_id not in self.categories_by_gender:
                self.categories_by_gender[gender_id] = []
            self.categories_by_gender[gender_id].append(i)

        # Create embeddings - Swedish only, use dictionary for translations
        texts = []
        for c in categories:
            # Use path (e.g., "Kläder > Jackor > Dunjackor") or just name
            text = c.get("path", c["name"]) or c["name"]
            # Enrich with dictionary translations
            text = self.dictionary.enrich_text(text)
            texts.append(text)

        logger.info(f"Category embedding examples: {texts[:3]}")
        logger.info(f"Categories by gender: {[(gid, len(idxs)) for gid, idxs in self.categories_by_gender.items()]}")
        self.category_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        # Pre-compute embeddings per gender for fast filtering
        for gender_id, indices in self.categories_by_gender.items():
            self.category_embeddings_by_gender[gender_id] = self.category_embeddings[indices]

    def set_local_genders(self, genders: list[dict]):
        """Set genders and create lookup."""
        self.local_genders = genders
        self.gender_lookup = {g["name"].lower(): g for g in genders}
        # Also create id lookup
        self.gender_id_lookup = {g["id"]: g for g in genders}

        # Simple embeddings - just the name
        texts = [g["name"] for g in genders]
        logger.info(f"Gender embeddings: {texts}")
        self.gender_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def _is_english(self, text: str) -> bool:
        """Check if text appears to be English (not Swedish).

        Uses simple heuristics - if text contains common English fashion terms
        or lacks Swedish characters, it's likely English.
        """
        if not text:
            return False

        text_lower = text.lower()

        # Common English fashion product types
        english_terms = {
            'jacket', 'coat', 'pants', 'trousers', 'shirt', 'dress', 'skirt',
            'sweater', 'hoodie', 'vest', 'cardigan', 'blouse', 'shorts',
            'jeans', 'leggings', 'tights', 'socks', 'underwear', 'bra',
            'shoes', 'boots', 'sneakers', 'sandals', 'hat', 'cap', 'gloves',
            'scarf', 'belt', 'bag', 'backpack', 'swimwear', 'bikini',
            't-shirt', 'polo', 'fleece', 'softshell', 'hardshell', 'parka',
            'blazer', 'suit', 'tie', 'top', 'bottom', 'outerwear', 'activewear'
        }

        # Check if any English term is in the text
        for term in english_terms:
            if term in text_lower:
                return True

        # Swedish-specific characters suggest it's Swedish
        swedish_chars = {'å', 'ä', 'ö'}
        if any(c in text_lower for c in swedish_chars):
            return False

        # If all ASCII and no Swedish indicators, assume English
        return text.isascii()

    def map_batch_colors(self, detected: list[str], csv_colors: list[str],
                          detected_confidences: list[float] = None) -> list[dict]:
        """Map batch of colors to local taxonomy.

        Returns dict with: id, name, confidence, suggestion
        - If confidence >= threshold: returns matched local color
        - If confidence < threshold: returns "unknown" + suggestion to create new color
        """
        if not self.local_colors:
            return [{"id": None, "name": "unknown", "confidence": 0.0, "suggestion": c} for c in csv_colors]

        if detected_confidences is None:
            detected_confidences = [1.0] * len(detected)

        # Minimum confidence to use CLIP color detection
        MIN_DETECTION_CONFIDENCE = 0.15

        # Build query: use CSV color primarily, only add detected if confident
        queries = []
        for d, c, conf in zip(detected, csv_colors, detected_confidences):
            parts = []
            # CSV color is usually reliable (e.g., "Red Illusion", "Evening Haze")
            if c:
                parts.append(c)
            # Only add detected color if CLIP was confident
            if d and conf >= MIN_DETECTION_CONFIDENCE:
                parts.append(d)
            queries.append(' '.join(parts) if parts else 'unknown')

        # Process in batches and use CPU for similarity to avoid OOM
        BATCH_SIZE = 512
        results = []
        color_cpu = self.color_embeddings.cpu()
        result_idx = 0

        for i in range(0, len(queries), BATCH_SIZE):
            batch_queries = queries[i:i + BATCH_SIZE]

            query_emb = self.model.encode(batch_queries, convert_to_tensor=True, show_progress_bar=False)
            query_cpu = query_emb.cpu()

            sims = torch.nn.functional.cosine_similarity(query_cpu.unsqueeze(1), color_cpu.unsqueeze(0), dim=2)
            best_idx = sims.argmax(dim=1)
            best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)

            for j, (idx, score) in enumerate(zip(best_idx, best_scores)):
                conf = round(score.item(), 3)
                csv_color = csv_colors[result_idx] if result_idx < len(csv_colors) else ''
                result_idx += 1

                if conf >= self.config.min_mapping_confidence:
                    results.append({
                        "id": self.local_colors[idx.item()]["id"],
                        "name": self.local_colors[idx.item()]["name"],
                        "confidence": conf,
                        "suggestion": None
                    })
                else:
                    results.append({
                        "id": None,
                        "name": "unknown",
                        "confidence": conf,
                        "suggestion": csv_color if csv_color else None
                })
        return results

    def map_batch_categories(self, product_types: list[str], titles: list[str],
                              descriptions: list[str], gender_ids: list[int]) -> list[dict]:
        """
        Map batch of categories to local taxonomy with GENDER FILTERING.

        Swedish-only matching. Categories are filtered by product gender before matching.
        Uses WikiDict dictionary for translations, no hardcoded translations.

        Returns dict with: id, name, confidence, suggested_category
        """
        if not self.local_categories:
            return [{"id": None, "name": None, "confidence": 0.0, "suggested_category": None} for _ in product_types]

        # Build queries - use GPT-SW3 to translate English product types to Swedish
        queries = []

        # Collect unique English product types for batch translation
        unique_english_types = list(set(p for p in product_types if p and self._is_english(p)))

        # Translate English product types to Swedish using GPT-SW3
        translation_map = {}
        if unique_english_types:
            logger.info(f"Translating {len(unique_english_types)} unique English product types to Swedish with GPT-SW3...")
            translated = self.translator.translate_batch(unique_english_types, src_lang='en')
            translation_map = dict(zip(unique_english_types, translated))
            logger.info(f"Translations: {list(translation_map.items())[:5]}...")

        for p, title, desc in zip(product_types, titles, descriptions):
            parts = []

            # 1. Product type - include both original AND Swedish translation
            if p:
                parts.append(p)
                # Add GPT-SW3 translation if available
                if p in translation_map and translation_map[p]:
                    parts.append(translation_map[p])

            # 2. Title keywords
            if title:
                parts.append(title)

            # 3. Description (often contains Swedish keywords)
            if desc:
                parts.append(desc[:100])

            # Build query and enrich with dictionary (adds word-level translations)
            query = ' '.join(parts) if parts else 'unknown'
            query = self.dictionary.enrich_text(query)
            queries.append(query)

        # Process with gender filtering
        BATCH_SIZE = 256
        results = []

        for i in range(0, len(queries), BATCH_SIZE):
            batch_queries = queries[i:i + BATCH_SIZE]
            batch_gender_ids = gender_ids[i:i + BATCH_SIZE]
            batch_product_types = product_types[i:i + BATCH_SIZE]
            batch_titles = titles[i:i + BATCH_SIZE]

            # Encode queries
            query_emb = self.model.encode(batch_queries, convert_to_tensor=True, show_progress_bar=False)

            # Process each product with its gender filter
            # FALLBACK THRESHOLD: If gender-filtered match is weak, try ALL categories
            GENDER_FALLBACK_THRESHOLD = 0.65

            for j, (q_emb, gender_id, pt, title) in enumerate(zip(query_emb, batch_gender_ids, batch_product_types, batch_titles)):
                best_idx = None
                best_score = 0.0

                # Step 1: Try gender-filtered categories first
                if gender_id in self.categories_by_gender:
                    cat_indices = self.categories_by_gender[gender_id]
                    cat_emb = self.category_embeddings_by_gender[gender_id]

                    sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), cat_emb, dim=1)
                    best_local_idx = sims.argmax().item()
                    best_score = sims[best_local_idx].item()
                    best_idx = cat_indices[best_local_idx]

                # Step 2: If gender match is weak OR gender not found, also try ALL categories
                if best_score < GENDER_FALLBACK_THRESHOLD or best_idx is None:
                    all_sims = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), self.category_embeddings, dim=1)
                    all_best_idx = all_sims.argmax().item()
                    all_best_score = all_sims[all_best_idx].item()

                    # Use the better match (gender-filtered OR all categories)
                    if all_best_score > best_score:
                        best_idx = all_best_idx
                        best_score = all_best_score

                conf = round(best_score, 3)

                # Safety check - should never happen but prevents crash
                if best_idx is None:
                    logger.warning(f"No category match found for product: {pt}")
                    results.append({
                        "id": None,
                        "name": "unknown",
                        "confidence": 0.0,
                        "suggested_category": pt if pt else "unknown"
                    })
                    continue

                if conf >= self.config.min_mapping_confidence:
                    results.append({
                        "id": self.local_categories[best_idx]["id"],
                        "name": self.local_categories[best_idx]["name"],
                        "confidence": conf,
                        "suggested_category": None
                    })
                else:
                    # Low confidence - suggest based on product type
                    suggestion = pt if pt else (title.split()[-1] if title else "unknown")
                    results.append({
                        "id": None,
                        "name": "unknown",
                        "confidence": conf,
                        "suggested_category": suggestion
                    })

        return results

    def map_batch_genders(self, csv_genders: list[str]) -> list[dict]:
        """Map batch of genders to local taxonomy."""
        if not self.local_genders:
            return [{"id": None, "name": None, "confidence": 0.0} for _ in csv_genders]

        queries = [g or "unknown" for g in csv_genders]
        query_emb = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), self.gender_embeddings.unsqueeze(0), dim=2)
        best_idx = sims.argmax(dim=1)
        best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        return [
            {"id": self.local_genders[idx.item()]["id"], "name": self.local_genders[idx.item()]["name"], "confidence": round(score.item(), 3)}
            for idx, score in zip(best_idx, best_scores)
        ]

    def generate_category_suggestion(self, detected_category: str, product_type: str,
                                      title: str, google_category: str = '') -> str:
        """
        Generate a smart category suggestion when no good local match exists.
        Uses CLIP detection + product context to suggest what category should be created.

        Priority:
        1. Product type (most specific from supplier)
        2. CLIP detected category (visual analysis)
        3. Title keywords
        4. Google category (fallback)

        Returns a normalized, clean category name suggestion.
        """
        suggestion_parts = []

        # 1. Normalize product_type if it's a known category
        if product_type:
            ptype_lower = product_type.lower().strip()
            # Check if it's a recognized category we can normalize
            for key, normalized in self.config.category_normalization.items():
                if key in ptype_lower:
                    # Found a match - use normalized version
                    # Add qualifiers from product_type if present
                    qualifiers = []
                    if 'running' in ptype_lower or 'athletic' in ptype_lower or 'sport' in ptype_lower:
                        qualifiers.append('Athletic')
                    if 'fleece' in ptype_lower:
                        qualifiers.append('Fleece')
                    if 'half zip' in ptype_lower or 'half-zip' in ptype_lower:
                        qualifiers.append('Half-Zip')
                    if 'crew' in ptype_lower:
                        qualifiers.append('Crew Neck')
                    if 'v-neck' in ptype_lower or 'vneck' in ptype_lower:
                        qualifiers.append('V-Neck')
                    if 'long sleeve' in ptype_lower:
                        qualifiers.append('Long Sleeve')
                    if 'short sleeve' in ptype_lower:
                        qualifiers.append('Short Sleeve')

                    if qualifiers:
                        return f"{' '.join(qualifiers)} {normalized}"
                    return normalized

            # If no normalization rule, clean up product_type
            suggestion_parts.append(product_type.strip().title())

        # 2. Use CLIP detected category if product_type was empty
        if not suggestion_parts and detected_category:
            detected_lower = detected_category.lower()
            normalized = self.config.category_normalization.get(detected_lower)
            if normalized:
                suggestion_parts.append(normalized)
            else:
                suggestion_parts.append(detected_category.title())

        # 3. Extract from title if still empty
        if not suggestion_parts and title:
            title_lower = title.lower()
            for key, normalized in self.config.category_normalization.items():
                if key in title_lower:
                    suggestion_parts.append(normalized)
                    break

        # 4. Fallback to Google category last segment
        if not suggestion_parts and google_category and '>' in google_category:
            last_segment = google_category.split('>')[-1].strip()
            if last_segment and last_segment.lower() not in ['clothing', 'apparel', 'accessories', 'other']:
                suggestion_parts.append(last_segment.title())

        # Return suggestion or None if we couldn't generate one
        if suggestion_parts:
            return suggestion_parts[0]  # Return first/best suggestion
        return None

    def nlp_classify_categories(self, titles: list[str], descriptions: list[str],
                                 product_types: list[str]) -> list[dict]:
        """
        BETA: NLP-based category classification using text only (no images).
        Uses title, description, and product_type to match against local categories.
        """
        if not self.local_categories:
            return [{"id": None, "name": None, "confidence": 0.0} for _ in titles]

        # Build rich text queries from all available text fields
        queries = []
        for title, desc, ptype in zip(titles, descriptions, product_types):
            parts = []
            if ptype:
                parts.append(ptype)
            if title:
                parts.append(title)
            if desc:
                # Take first 100 chars of description
                parts.append(desc[:100])
            queries.append(' '.join(parts) if parts else 'unknown')

        # Encode and match
        query_emb = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), self.category_embeddings.unsqueeze(0), dim=2)
        best_idx = sims.argmax(dim=1)
        best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        # Return results, but mark as "unknown" if confidence too low
        results = []
        for idx, score in zip(best_idx, best_scores):
            conf = round(score.item(), 3)
            if conf >= self.config.min_mapping_confidence:
                results.append({
                    "id": self.local_categories[idx.item()]["id"],
                    "name": self.local_categories[idx.item()]["name"],
                    "confidence": conf
                })
            else:
                results.append({"id": None, "name": "unknown", "confidence": conf})
        return results


# =============================================================================
# CATEGORY CLASSIFIER - Zero-shot with BART
# =============================================================================

class CategoryClassifier:
    """Zero-shot classifier for product categories using BART.

    Uses actual database category names as labels for proper classification.
    Much more accurate than sentence similarity matching.
    """

    def __init__(self, config: Config, classifier_pipeline):
        self.config = config
        self.classifier = classifier_pipeline
        self.category_labels = []
        self.category_map = {}  # label -> {id, name, path}

    def set_categories(self, categories: list[dict]):
        """Set available categories from database.

        Only use level 2-3 categories to keep label count manageable.
        830 labels is too slow for zero-shot classification.
        """
        # Filter to level 2-3 categories only (main categories, not too deep)
        filtered = [cat for cat in categories if cat.get('level', 99) in [2, 3]]

        # Group by name to avoid duplicates, prefer higher level (more general)
        seen_names = {}
        for cat in filtered:
            name = cat['name']
            if name not in seen_names or cat.get('level', 99) < seen_names[name].get('level', 99):
                seen_names[name] = cat

        self.category_map = {cat['name']: cat for cat in seen_names.values()}
        self.category_labels = list(self.category_map.keys())
        logger.info(f"CategoryClassifier loaded {len(self.category_labels)} category labels (level 2-3 only, was 830)")

    def classify_batch(self, contexts: list[dict]) -> list[dict]:
        """
        Classify products into categories using zero-shot classification.
        Returns category match + suggestion if no good match.
        """
        if not self.category_labels:
            return [{"id": None, "name": "unknown", "confidence": 0.0, "suggestion": None} for _ in contexts]

        # Build text for classification
        texts = []
        for ctx in contexts:
            parts = []
            if ctx.get('product_type'):
                parts.append(ctx['product_type'])
            if ctx.get('title'):
                parts.append(ctx['title'])
            if ctx.get('description'):
                parts.append(ctx['description'][:150])
            texts.append(' '.join(parts) if parts else 'unknown product')

        # Run classification with batching
        logger.info(f"Classifying {len(texts)} products into {len(self.category_labels)} categories...")
        outputs = list(self.classifier(
            texts,
            self.category_labels,
            batch_size=self.config.bart_batch_size
        ))

        # Debug: show first result format
        if outputs:
            logger.debug(f"Category classifier output format: {type(outputs[0])}, sample: {outputs[0]}")

        # Build results - handle both dict and list output formats
        results = []
        for i, out in enumerate(outputs):
            # Handle different output formats
            if isinstance(out, dict):
                best_label = out['labels'][0]
                best_score = out['scores'][0]
            elif isinstance(out, list) and len(out) > 0:
                best_label = out[0].get('label', out[0].get('labels', ['unknown'])[0])
                best_score = out[0].get('score', out[0].get('scores', [0.0])[0])
            else:
                logger.warning(f"Unexpected category classifier output: {type(out)}: {out}")
                results.append({"id": None, "name": "unknown", "confidence": 0.0, "suggestion": None})
                continue

            cat_info = self.category_map.get(best_label, {})

            # If confidence too low, suggest creating a new category
            suggestion = None
            if best_score < self.config.min_mapping_confidence:
                ptype = contexts[i].get('product_type', '')
                if ptype:
                    suggestion = ptype

            results.append({
                "id": cat_info.get('id') if best_score >= self.config.min_mapping_confidence else None,
                "name": best_label if best_score >= self.config.min_mapping_confidence else "unknown",
                "confidence": round(best_score, 3),
                "suggestion": suggestion
            })

        return results


# =============================================================================
# LLM CATEGORY CLASSIFIER - Uses local LLM for accurate understanding
# =============================================================================

class LLMCategoryClassifier:
    """Uses a local LLM to understand products and match to categories.

    This solves the problem of embedding similarity not understanding semantic meaning.
    The LLM reads the product info and determines the best category match.

    Uses vLLM for fast, stable inference (2-4x faster than transformers).
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.local_categories = []
        self.category_list_str = ""

    def load_model(self):
        """Load the LLM model using vLLM or transformers."""
        if self.model is not None:
            return

        if self.config.use_vllm:
            try:
                from vllm import LLM, SamplingParams
                logger.info(f"Loading LLM with vLLM: {self.config.llm_model}...")
                self.model = LLM(
                    model=self.config.llm_model,
                    dtype="half",
                    gpu_memory_utilization=0.5,  # Leave room for other models
                    trust_remote_code=True,
                )
                self.sampling_params = SamplingParams(
                    temperature=0.1,
                    max_tokens=50,
                )
                self.use_vllm = True
                logger.info("vLLM loaded successfully!")
                return
            except ImportError:
                logger.warning("vLLM not installed, falling back to transformers")
            except Exception as e:
                logger.warning(f"vLLM failed: {e}, falling back to transformers")

        # Fallback to transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"Loading LLM with transformers: {self.config.llm_model}...")
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
        logger.info("LLM loaded successfully")

    def set_categories(self, categories: list[dict]):
        """Set available categories from database."""
        self.local_categories = categories
        # Build category list - limit to avoid prompt overflow
        # Group unique names and create embeddings for semantic search
        cat_names = sorted(set(c["name"] for c in categories))
        self.all_category_names = cat_names

        # For LLM prompt, limit to 200 most common/important categories
        # The LLM will match semantically, so we don't need ALL names
        MAX_CATEGORIES_IN_PROMPT = 200
        self.category_list_str = ", ".join(cat_names[:MAX_CATEGORIES_IN_PROMPT])
        if len(cat_names) > MAX_CATEGORIES_IN_PROMPT:
            self.category_list_str += f" (and {len(cat_names) - MAX_CATEGORIES_IN_PROMPT} more)"

        logger.info(f"LLM Classifier: {len(cat_names)} categories, {min(len(cat_names), MAX_CATEGORIES_IN_PROMPT)} in prompt")

    def classify_batch(self, contexts: list[dict]) -> list[dict]:
        """
        Use LLM to classify products into categories.

        For each product, the LLM:
        1. Reads title, description, product_type
        2. Understands what the product IS
        3. Matches to best local category OR suggests a new one
        """
        if not self.config.use_llm_category:
            return [{"id": None, "name": None, "confidence": 0.0, "suggested_category": None} for _ in contexts]

        self.load_model()

        # Build category lookup
        cat_lookup = {c["name"].lower(): c for c in self.local_categories}

        # Build all prompts
        prompts = [self._build_prompt(ctx) for ctx in contexts]

        if self.use_vllm:
            # vLLM - process all at once (much faster!)
            logger.info(f"vLLM generating {len(prompts)} responses...")
            outputs = self.model.generate(prompts, self.sampling_params)

            results = []
            for output, ctx in zip(outputs, contexts):
                answer = output.outputs[0].text.strip()
                result = self._parse_response(answer, cat_lookup, ctx)
                results.append(result)
            return results

        else:
            # Transformers fallback - batch processing
            from tqdm import tqdm
            results = []
            batch_size = 8

            for i in tqdm(range(0, len(contexts), batch_size), desc="LLM classifying", unit="batch"):
                batch_prompts = prompts[i:i + batch_size]
                batch_contexts = contexts[i:i + batch_size]

                # Tokenize all at once
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

                # Generate all at once
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode all responses
                responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Parse each response
                for j, (response, ctx) in enumerate(zip(responses, batch_contexts)):
                    prompt = batch_prompts[j]
                    answer = response[len(prompt):].strip()
                    result = self._parse_response(answer, cat_lookup, ctx)
                    results.append(result)

            return results

    def _build_prompt(self, ctx: dict) -> str:
        """Build prompt for LLM category classification."""
        product_type = ctx.get('product_type', '')
        title = ctx.get('title', '')
        description = ctx.get('description', '')[:300]

        prompt = f"""You are a Swedish e-commerce product categorization expert. Match products to Swedish category names.

Product Information:
- Type: {product_type}
- Title: {title}
- Description: {description}

Available Categories (Swedish): {self.category_list_str}

IMPORTANT: Categories are in Swedish. Common translations:
- Jacket/Coat = Jacka, Ytterplagg, Skaljacka
- Pants/Trousers = Byxor, Skalbyxor
- Shirt = Skjorta, Tröja
- Dress = Klänning
- Shoes = Skor
- Boots = Stövlar, Kängor

Instructions:
1. Understand what this product IS (jacket, pants, shoes, etc.)
2. Find the BEST matching Swedish category from the list
3. Match semantically - "Jacket" should match "Jacka" or "Ytterplagg"

Respond in this exact format:
CATEGORY: [exact Swedish category name from the list]
CONFIDENCE: [high/medium/low]
SUGGESTION: [new category name if no good match, otherwise "none"]

Response:"""
        return prompt

    def _parse_response(self, answer: str, cat_lookup: dict, ctx: dict) -> dict:
        """Parse LLM response into structured result. More robust parsing."""
        lines = answer.strip().split('\n')

        category_name = None
        confidence = 0.5
        suggestion = None

        for line in lines:
            line = line.strip()
            line_lower = line.lower()

            # More robust parsing - handle variations like "Category:", "CATEGORY:", "category :"
            if line_lower.startswith('category'):
                # Extract after the colon
                if ':' in line:
                    category_name = line.split(':', 1)[1].strip()
            elif line_lower.startswith('confidence'):
                if ':' in line:
                    conf_str = line.split(':', 1)[1].strip().lower()
                    if 'high' in conf_str:
                        confidence = 0.9
                    elif 'medium' in conf_str or 'med' in conf_str:
                        confidence = 0.7
                    else:
                        confidence = 0.4
            elif line_lower.startswith('suggestion'):
                if ':' in line:
                    sugg = line.split(':', 1)[1].strip()
                    if sugg.lower() not in ['none', 'n/a', '-', '']:
                        suggestion = sugg

        # Find matching category
        cat_info = cat_lookup.get(category_name.lower() if category_name else '', {})

        if cat_info and confidence >= 0.5:
            return {
                "id": cat_info.get("id"),
                "name": cat_info.get("name"),
                "confidence": confidence,
                "suggested_category": None
            }
        else:
            return {
                "id": None,
                "name": "unknown",
                "confidence": confidence,
                "suggested_category": suggestion or ctx.get('product_type')
            }


# =============================================================================
# SIZE TYPE CLASSIFIER - Runs in parallel
# =============================================================================

class SizeTypeClassifier:
    """Zero-shot classifier for size type and product understanding."""

    # High-level product type labels for NLP understanding
    PRODUCT_TYPE_LABELS = [
        # Outerwear
        "ski jacket / shell jacket / ski wear",
        "ski pants / shell pants / snow pants",
        "rain jacket / waterproof jacket",
        "winter jacket / down jacket / puffer",
        "fleece jacket / fleece sweater",
        "softshell jacket",
        "windbreaker",
        # Tops
        "t-shirt / tee",
        "long sleeve shirt / dress shirt",
        "sweater / pullover / knit",
        "hoodie / sweatshirt",
        "polo shirt",
        "tank top / singlet",
        "blouse",
        "base layer top / thermal top",
        # Bottoms
        "jeans / denim pants",
        "casual pants / chinos",
        "sweatpants / joggers",
        "shorts",
        "hiking pants / outdoor pants",
        "leggings / tights",
        "base layer pants / thermal pants",
        # Dresses & Skirts
        "dress",
        "skirt",
        # Underwear & Socks
        "underwear / briefs / boxers",
        "bra / sports bra",
        "socks",
        "base layer / thermal underwear set",
        # Footwear
        "sneakers / athletic shoes",
        "boots / hiking boots",
        "sandals",
        "dress shoes",
        # Accessories
        "hat / beanie / cap",
        "gloves / mittens",
        "scarf",
        "belt",
        "bag / backpack",
        # Swimwear
        "swimwear / bikini / swim trunks",
    ]

    def __init__(self, config: Config):
        self.config = config
        self.device = 0 if torch.cuda.is_available() else -1

        # Load with optimizations
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config.zero_shot_model,
            device=self.device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            torch_dtype=torch.float16,
        )

        # Compile model for 20-40% speedup (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.cuda.is_available():
            try:
                self.classifier.model = torch.compile(self.classifier.model, mode="reduce-overhead")
                logger.info("Zero-shot model compiled with torch.compile() for faster inference")
            except Exception as e:
                logger.warning(f"torch.compile() failed (will use uncompiled): {e}")

    def understand_products(self, contexts: list[dict]) -> list[dict]:
        """
        Use NLP to understand what each product actually IS.
        Returns high-level product type understanding with confidence.

        This helps with cases like:
        - "Skalbyxor" → "ski pants / shell pants"
        - "Underställ" → "base layer / thermal underwear"
        - "Klänning" → "dress"
        """
        texts = []
        for ctx in contexts:
            parts = []
            if ctx.get('product_type'):
                parts.append(f"Product: {ctx['product_type']}")
            if ctx.get('title'):
                parts.append(f"Title: {ctx['title']}")
            if ctx.get('description'):
                parts.append(f"Description: {ctx['description'][:200]}")
            texts.append(' '.join(parts) if parts else 'unknown product')

        if not texts:
            return [{"product_type_nlp": None, "nlp_confidence": 0.0} for _ in contexts]

        # Run zero-shot classification
        logger.info(f"NLP understanding {len(texts)} products...")
        results_list = list(self.classifier(
            texts,
            self.PRODUCT_TYPE_LABELS,
            batch_size=self.config.bart_batch_size,
            multi_label=False
        ))

        results = []
        for out in results_list:
            if isinstance(out, dict):
                results.append({
                    "product_type_nlp": out["labels"][0],
                    "nlp_confidence": round(out["scores"][0], 3)
                })
            elif isinstance(out, list) and len(out) > 0:
                results.append({
                    "product_type_nlp": out[0].get("label", out[0].get("labels", ["unknown"])[0]),
                    "nlp_confidence": round(out[0].get("score", out[0].get("scores", [0.0])[0]), 3)
                })
            else:
                results.append({"product_type_nlp": None, "nlp_confidence": 0.0})

        return results

    def classify_all(self, contexts: list[dict]) -> list[dict]:
        """
        Classify size type using available context.
        Priority: product_type > title > description
        (google_product_category often too generic or empty)
        """
        valid_texts = []
        valid_indices = []

        for i, ctx in enumerate(contexts):
            parts = []

            # 1. Product type is most reliable (e.g., "Half zip sweater", "Jeans", "Sneakers")
            if ctx.get('product_type'):
                parts.append(f"This is a {ctx['product_type']}.")

            # 2. Title often contains product type info
            if ctx.get('title'):
                parts.append(ctx['title'])

            # 3. Description as fallback context
            if ctx.get('description'):
                parts.append(ctx['description'][:150])

            text = " ".join(parts)

            if text.strip():
                valid_texts.append(text[:512])
                valid_indices.append(i)

        if not valid_texts:
            return [{"size_type": None, "confidence": 0.0} for _ in contexts]

        # Process with batching - pass list directly to pipeline
        results_list = list(self.classifier(
            valid_texts,
            self.config.size_type_labels,
            batch_size=self.config.bart_batch_size
        ))

        # Debug: show first result format
        if results_list:
            logger.debug(f"Size classifier output format: {type(results_list[0])}, sample: {results_list[0]}")

        # Map back
        results = [{"size_type": None, "confidence": 0.0} for _ in contexts]
        for i, valid_idx in enumerate(valid_indices):
            out = results_list[i]
            # Handle both dict and nested list formats
            if isinstance(out, dict):
                results[valid_idx] = {
                    "size_type": out["labels"][0],
                    "confidence": round(out["scores"][0], 3)
                }
            elif isinstance(out, list) and len(out) > 0:
                # Pipeline sometimes returns list of dicts
                results[valid_idx] = {
                    "size_type": out[0]["label"] if "label" in out[0] else out[0]["labels"][0],
                    "confidence": round(out[0]["score"] if "score" in out[0] else out[0]["scores"][0], 3)
                }
            else:
                logger.warning(f"Unexpected size classifier output: {type(out)}: {out}")
        return results


# =============================================================================
# MAIN PIPELINE - PARALLEL EXECUTION
# =============================================================================

class ProductEnrichmentPipeline:
    """Main pipeline with parallel execution for maximum speed."""

    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config)

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("INITIALIZING PIPELINE (Optimized for A10)")
        logger.info("=" * 60)

        await self.db.connect()

        # Fetch taxonomies in parallel
        colors, categories, genders = await asyncio.gather(
            self.db.fetch_colors(),
            self.db.fetch_categories(),
            self.db.fetch_genders()
        )

        logger.info(f"Loaded {len(colors)} colors, {len(categories)} categories, {len(genders)} genders")

        # Initialize models
        logger.info("Loading FashionSigLIP model...")
        self.fashion = FashionImageProcessor(self.config)
        self.fashion.set_local_colors(colors)  # Use YOUR database colors for detection!

        logger.info("Loading Sentence Transformer...")
        self.mapper = TaxonomyMapper(self.config)
        self.mapper.set_local_colors(colors)
        self.mapper.set_local_categories(categories)
        self.mapper.set_local_genders(genders)

        logger.info("Loading BART classifier...")
        self.size_classifier = SizeTypeClassifier(self.config)

        # Share BART pipeline for category classification (saves memory)
        logger.info("Setting up Category classifier (zero-shot)...")
        self.category_classifier = CategoryClassifier(self.config, self.size_classifier.classifier)
        self.category_classifier.set_categories(categories)

        # LLM-based category classifier (optional, for best accuracy)
        self.llm_classifier = None
        if self.config.use_llm_category:
            logger.info("Setting up LLM Category classifier...")
            self.llm_classifier = LLMCategoryClassifier(self.config)
            self.llm_classifier.set_categories(categories)

        logger.info("All models loaded!")

    async def cleanup(self):
        await self.db.close()

    def read_csv(self, path: str) -> list[dict]:
        logger.info(f"Reading CSV: {path}")
        products = []
        with open(path, 'r', encoding='iso-8859-1') as f:
            reader = csv.DictReader(f, delimiter=';', quotechar='"')
            for row in reader:
                products.append(row)
        logger.info(f"Loaded {len(products)} products")
        return products

    def write_csv(self, products: list[dict], path: str):
        logger.info(f"Writing CSV: {path}")
        if not products:
            return
        fieldnames = list(products[0].keys())
        with open(path, 'w', encoding='iso-8859-1', newline='', errors='replace') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(products)

    async def process_images_batch(self, products: list[dict], downloader: FastImageDownloader) -> list[dict]:
        """Process a batch: download images + CLIP inference."""
        urls = [p.get('image_link', '') for p in products]

        # Download images
        images = await downloader.download_batch(urls)

        # CLIP inference
        clip_results = self.fashion.process_batch(images)

        return clip_results

    def _process_text_categories(self, product_types: list[str], titles: list[str],
                                  descriptions: list[str], csv_genders: list[str],
                                  size_contexts: list[dict], item_group_ids: list[str]) -> tuple[list[dict], list[dict]]:
        """Process categories using TEXT only with GENDER FILTERING.

        OPTIMIZATION: Deduplicates by item_group_id - variants of same product share category.
        Returns (mapped_categories, llm_categories) for ALL products.
        """
        total_products = len(product_types)

        # First, map genders to get gender_ids for filtering
        logger.info("Mapping genders for category filtering...")
        mapped_genders = self.mapper.map_batch_genders(csv_genders)
        gender_ids = [g["id"] for g in mapped_genders]

        # =====================================================================
        # DEDUPLICATION: Group by item_group_id, process only unique groups
        # =====================================================================
        group_to_indices = {}  # item_group_id -> list of product indices
        unique_indices = []     # First index of each unique group

        for i, group_id in enumerate(item_group_ids):
            # Use group_id if available, otherwise treat each product as unique
            key = group_id if group_id else f"__unique_{i}"
            if key not in group_to_indices:
                group_to_indices[key] = []
                unique_indices.append(i)  # First product in this group
            group_to_indices[key].append(i)

        num_unique = len(unique_indices)
        logger.info(f"Category dedup: {total_products} products -> {num_unique} unique groups ({100*num_unique/total_products:.1f}%)")

        # Extract data for unique products only
        unique_product_types = [product_types[i] for i in unique_indices]
        unique_titles = [titles[i] for i in unique_indices]
        unique_descriptions = [descriptions[i] for i in unique_indices]
        unique_gender_ids = [gender_ids[i] for i in unique_indices]
        unique_contexts = [size_contexts[i] for i in unique_indices]

        # Embedding-based category matching with gender filtering
        logger.info(f"Running text-based category embeddings on {num_unique} unique groups (with gender filter)...")
        unique_mapped = self.mapper.map_batch_categories(
            unique_product_types, unique_titles, unique_descriptions, unique_gender_ids
        )

        # LLM for low-confidence matches - on UNIQUE products only
        unique_llm = [None] * num_unique
        if self.config.use_llm_category and self.llm_classifier:
            low_conf_unique_indices = [
                i for i, cat in enumerate(unique_mapped)
                if cat["confidence"] < self.config.llm_confidence_threshold
            ]

            if low_conf_unique_indices:
                logger.info(f"Running LLM on {len(low_conf_unique_indices)}/{num_unique} low-confidence groups...")
                low_conf_contexts = [unique_contexts[i] for i in low_conf_unique_indices]
                llm_results = self.llm_classifier.classify_batch(low_conf_contexts)

                for idx, result in zip(low_conf_unique_indices, llm_results):
                    unique_llm[idx] = result
                logger.info("LLM classification complete!")
            else:
                logger.info("All groups matched with high confidence - skipping LLM")

        # =====================================================================
        # EXPAND: Map results back to ALL products (variants share category)
        # =====================================================================
        mapped_categories = [None] * total_products
        llm_categories = [None] * total_products

        for unique_idx, original_idx in enumerate(unique_indices):
            # Get the group_id for this unique product
            group_id = item_group_ids[original_idx] if item_group_ids[original_idx] else f"__unique_{original_idx}"

            # Apply result to ALL products in this group
            for product_idx in group_to_indices[group_id]:
                mapped_categories[product_idx] = unique_mapped[unique_idx]
                llm_categories[product_idx] = unique_llm[unique_idx]

        return mapped_categories, llm_categories

    async def run(self, input_path: str, output_path: str):
        """Run the full pipeline with PARALLEL execution for maximum speed."""
        start_time = time.time()

        try:
            await self.initialize()

            # Read input
            products = self.read_csv(input_path)
            total = len(products)
            batch_size = self.config.batch_size
            num_batches = (total + batch_size - 1) // batch_size

            logger.info("=" * 60)
            logger.info(f"PROCESSING {total} PRODUCTS (PARALLEL MODE)")
            logger.info(f"Batch size: {batch_size}, Total batches: {num_batches}")
            logger.info("=" * 60)

            # Extract text data for parallel processing
            csv_colors = [p.get('color', '') for p in products]
            csv_genders = [p.get('gender', '') for p in products]
            product_types = [p.get('product_type', '') for p in products]
            titles = [p.get('title', '') for p in products]
            descriptions = [p.get('description', '') for p in products]
            item_group_ids = [p.get('item_group_id', '') for p in products]  # For deduplication

            # Build rich context for classification
            size_contexts = [
                {
                    'title': p.get('title', ''),
                    'description': p.get('description', ''),
                    'product_type': p.get('product_type', ''),
                    'google_product_category': p.get('google_product_category', ''),
                }
                for p in products
            ]

            # =====================================================================
            # PARALLEL EXECUTION: Run TEXT and IMAGE processing simultaneously
            # =====================================================================

            # PARALLEL TASK 1: Size type classification (TEXT only)
            logger.info("Starting PARALLEL tasks...")
            logger.info("  -> Task 1: Size classification (text)")
            size_task = asyncio.create_task(
                asyncio.to_thread(self.size_classifier.classify_all, size_contexts)
            )

            # PARALLEL TASK 2: Category embeddings + LLM (TEXT only - DEDUPLICATED by item_group_id!)
            logger.info("  -> Task 2: Category embeddings + LLM (text, deduplicated)")
            category_task = asyncio.create_task(
                asyncio.to_thread(
                    self._process_text_categories,
                    product_types, titles, descriptions, csv_genders, size_contexts, item_group_ids
                )
            )

            # PARALLEL TASK 3: Zero-shot category (OPTIONAL)
            zeroshot_task = None
            if self.config.use_zeroshot_category:
                logger.info("  -> Task 3: Zero-shot classification (text)")
                zeroshot_task = asyncio.create_task(
                    asyncio.to_thread(self.category_classifier.classify_batch, size_contexts)
                )

            # PARALLEL TASK 4: Download + CLIP for COLORS (IMAGE-based)
            logger.info("  -> Task 4: Image download + CLIP colors")
            all_urls = [p.get('image_link', '') for p in products]

            async with FastImageDownloader(self.config) as downloader:
                all_images = await downloader.download_batch(all_urls)
                logger.info(f"Image downloads: {downloader.stats['success']} success, {downloader.stats['failed']} failed")

            # Process images on GPU for COLOR detection only
            logger.info("Processing images on GPU (color detection)...")
            all_clip_results = []
            with tqdm(total=num_batches, desc="GPU colors", unit="batch") as pbar:
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, total)
                    batch_images = all_images[start_idx:end_idx]
                    clip_results = self.fashion.process_batch(batch_images)
                    all_clip_results.extend(clip_results)
                    pbar.set_postfix({"products": f"{end_idx}/{total}"})
                    pbar.update(1)

            # =====================================================================
            # Wait for all parallel tasks to complete
            # =====================================================================
            logger.info("Waiting for parallel tasks to complete...")

            size_results = await size_task
            logger.info("  -> Size classification: DONE")

            mapped_categories, llm_categories = await category_task
            logger.info("  -> Category embeddings + LLM: DONE")

            if zeroshot_task:
                zeroshot_categories = await zeroshot_task
                logger.info("  -> Zero-shot classification: DONE")
            else:
                zeroshot_categories = [{"id": None, "name": None, "confidence": 0.0, "suggestion": None} for _ in products]

            logger.info("All parallel tasks complete!")

            # Extract color values from CLIP (images used for COLOR only)
            detected_colors = [r["color"] for r in all_clip_results]
            detected_color_ids = [r["color_id"] for r in all_clip_results]
            detected_color_confidences = [r["color_conf"] for r in all_clip_results]
            detected_categories = [r["category"] for r in all_clip_results]  # Kept for reference only
            detected_category_confidences = [r["category_conf"] for r in all_clip_results]

            # Free GPU memory
            torch.cuda.empty_cache()

            # Map colors and genders (fast operations)
            logger.info("Mapping colors and genders...")
            mapped_colors = self.mapper.map_batch_colors(
                detected_colors, csv_colors, detected_color_confidences
            )
            mapped_genders = self.mapper.map_batch_genders(csv_genders)

            # Enrich products
            logger.info("Enriching products...")
            for i, product in enumerate(products):
                # Color - CLIP now detects directly into YOUR local colors!
                product['detected_color'] = detected_colors[i]
                product['detected_color_id'] = detected_color_ids[i]  # Direct from CLIP using your colors
                product['detected_color_confidence'] = detected_color_confidences[i]
                # Also keep sentence-similarity mapping for comparison
                product['mapped_local_color_id'] = mapped_colors[i]["id"]
                product['mapped_local_color'] = mapped_colors[i]["name"]
                product['color_mapping_confidence'] = mapped_colors[i]["confidence"]
                product['suggestion_create_color_name'] = mapped_colors[i]["suggestion"]

                # Category (image-based + similarity matching - kept for comparison)
                product['detected_category'] = detected_categories[i]
                product['detected_category_confidence'] = all_clip_results[i]["category_conf"]
                product['mapped_local_category_id'] = mapped_categories[i]["id"]
                product['mapped_local_category'] = mapped_categories[i]["name"]
                product['category_mapping_confidence'] = mapped_categories[i]["confidence"]
                # AI-suggested category when no good local match exists
                product['suggested_category_name'] = mapped_categories[i].get("suggested_category")

                # LLM-based category (most accurate - uses NLU to understand product)
                llm_result = llm_categories[i] if llm_categories and llm_categories[i] else None
                if llm_result:
                    product['llm_category_id'] = llm_result["id"]
                    product['llm_category'] = llm_result["name"]
                    product['llm_category_confidence'] = llm_result["confidence"]
                    product['llm_suggested_category'] = llm_result.get("suggested_category")
                    # Use LLM as primary if has result
                    if llm_result["id"]:
                        product['best_category_id'] = llm_result["id"]
                        product['best_category'] = llm_result["name"]
                        product['best_category_source'] = "llm"
                    else:
                        product['best_category_id'] = mapped_categories[i]["id"]
                        product['best_category'] = mapped_categories[i]["name"]
                        product['best_category_source'] = "embedding"
                else:
                    # High confidence embedding match - no LLM needed
                    product['llm_category_id'] = None
                    product['llm_category'] = None
                    product['llm_category_confidence'] = None
                    product['llm_suggested_category'] = None
                    product['best_category_id'] = mapped_categories[i]["id"]
                    product['best_category'] = mapped_categories[i]["name"]
                    product['best_category_source'] = "embedding_high_conf"

                # Size type
                product['size_type'] = size_results[i]["size_type"]
                product['size_type_confidence'] = size_results[i]["confidence"]

                # Gender
                product['mapped_local_gender_id'] = mapped_genders[i]["id"]
                product['mapped_local_gender'] = mapped_genders[i]["name"]
                product['gender_mapping_confidence'] = mapped_genders[i]["confidence"]

                # Zero-shot category classification (only if enabled)
                if self.config.use_zeroshot_category:
                    product['zeroshot_category_id'] = zeroshot_categories[i]["id"]
                    product['zeroshot_category'] = zeroshot_categories[i]["name"]
                    product['zeroshot_category_confidence'] = zeroshot_categories[i]["confidence"]
                    product['suggestion_create_category_name'] = zeroshot_categories[i]["suggestion"]
                else:
                    # Use sentence similarity results as primary when zero-shot disabled
                    product['zeroshot_category_id'] = mapped_categories[i]["id"]
                    product['zeroshot_category'] = mapped_categories[i]["name"]
                    product['zeroshot_category_confidence'] = mapped_categories[i]["confidence"]
                    # Use the smart AI-generated suggestion instead of just product_type
                    product['suggestion_create_category_name'] = mapped_categories[i].get("suggested_category")

            # Write output
            self.write_csv(products, output_path)

            # Final stats
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0

            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED")
            logger.info(f"Total products: {total}")
            logger.info(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
            logger.info(f"Processing rate: {rate:.1f} products/second")
            logger.info(f"Output: {output_path}")
            logger.info("=" * 60)

        finally:
            await self.cleanup()


# =============================================================================
# CONFIGURATION - HARDCODED PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "outputs"
INPUT_FILE = INPUT_DIR / "products.csv"
OUTPUT_FILE = OUTPUT_DIR / "enriched_products.csv"


# =============================================================================
# ENTRY POINT
# =============================================================================

async def main():
    """Main entry point."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        sys.exit(1)

    logger.info(f"Input: {INPUT_FILE}")
    logger.info(f"Output: {OUTPUT_FILE}")

    config = Config()
    pipeline = ProductEnrichmentPipeline(config)
    await pipeline.run(str(INPUT_FILE), str(OUTPUT_FILE))


if __name__ == "__main__":
    asyncio.run(main())
