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
from datasets import Dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
import open_clip
from transformers import pipeline
from sentence_transformers import SentenceTransformer

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
    """Pipeline configuration - optimized for A10 GPU."""
    # Database
    db_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', ''))

    # Processing - MAXIMIZED for A10
    batch_size: int = 256  # Increased from 128
    max_concurrent_downloads: int = 200  # Increased from 100
    download_timeout: int = 15  # Reduced timeout for faster failure
    use_fp16: bool = True
    num_workers: int = 8  # For DataLoader

    # Models
    # Fashion-specific model - 57% better than generic CLIP for clothing
    fashion_model: str = "Marqo/marqo-fashionSigLIP"
    # Multilingual model for Swedish categories + English queries
    sentence_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    zero_shot_model: str = "facebook/bart-large-mnli"

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
        "jeans/pants", "shirt/top", "dress", "shoes", "socks",
        "underwear", "jacket/coat", "accessories", "swimwear", "one-size"
    ])

    def get_db_url(self) -> str:
        if self.db_url:
            return self.db_url
        return ""


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
            min_size=5,
            max_size=20
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

    57% better recall than generic CLIP for fashion/clothing detection.
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

        # Pre-compute text embeddings for colors and categories
        self._precompute_embeddings()

    @torch.no_grad()
    def _precompute_embeddings(self):
        """Pre-compute all text embeddings once for fast inference."""
        # Color prompts - fashion-specific phrasing
        color_texts = [f"{c} colored clothing" for c in self.config.color_prompts]
        color_tokens = self.tokenizer(color_texts).to(self.device)
        self.color_embeddings = self.model.encode_text(color_tokens, normalize=True)
        if self.config.use_fp16:
            self.color_embeddings = self.color_embeddings.half()

        # Category prompts - fashion-specific phrasing
        category_texts = [f"a {c}" for c in self.config.category_prompts]
        category_tokens = self.tokenizer(category_texts).to(self.device)
        self.category_embeddings = self.model.encode_text(category_tokens, normalize=True)
        if self.config.use_fp16:
            self.category_embeddings = self.category_embeddings.half()

        logger.info(f"Pre-computed embeddings for {len(self.config.color_prompts)} colors, {len(self.config.category_prompts)} categories")

    @torch.no_grad()
    def process_batch(self, images: list[Image.Image]) -> list[dict]:
        """Process batch of images - returns color and category predictions."""
        if not images:
            return []

        # Filter valid images
        valid_indices = [i for i, img in enumerate(images) if img is not None]
        valid_images = [images[i] for i in valid_indices]

        if not valid_images:
            return [{"color": None, "category": None, "color_conf": 0, "category_conf": 0} for _ in images]

        # Preprocess and stack images
        processed_images = torch.stack([self.preprocess(img) for img in valid_images]).to(self.device)
        if self.config.use_fp16:
            processed_images = processed_images.half()

        # Get image features (normalized)
        image_features = self.model.encode_image(processed_images, normalize=True)

        # Compute similarities (scale by 100 as per FashionSigLIP convention)
        color_sims = (100.0 * image_features @ self.color_embeddings.T).softmax(dim=-1)
        category_sims = (100.0 * image_features @ self.category_embeddings.T).softmax(dim=-1)

        color_probs, color_idx = color_sims.max(dim=-1)
        category_probs, category_idx = category_sims.max(dim=-1)

        # Build results
        results = [{"color": None, "category": None, "color_conf": 0.0, "category_conf": 0.0} for _ in images]
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = {
                "color": self.config.color_prompts[color_idx[i].item()],
                "category": self.config.category_prompts[category_idx[i].item()],
                "color_conf": round(color_probs[i].item(), 3),
                "category_conf": round(category_probs[i].item(), 3)
            }
        return results


# =============================================================================
# TAXONOMY MAPPER - Uses sentence embeddings
# =============================================================================

class TaxonomyMapper:
    """Maps values to local taxonomy using sentence embeddings."""

    # Swedish → English translations for common clothing categories
    SWEDISH_TO_ENGLISH = {
        'jackor': 'jackets coats outerwear',
        'byxor': 'pants trousers',
        'jeans': 'jeans denim',
        'shorts': 'shorts',
        'klänningar': 'dresses',
        'kjolar': 'skirts',
        'toppar': 'tops blouses shirts',
        't-shirts': 't-shirts tees',
        'skjortor': 'shirts button-up',
        'tröjor': 'sweaters knitwear pullovers',
        'sweatshirts': 'sweatshirts hoodies',
        'kavajer': 'blazers suit jackets',
        'kostymer': 'suits',
        'underkläder': 'underwear lingerie',
        'strumpor': 'socks hosiery',
        'skor': 'shoes footwear',
        'sneakers': 'sneakers trainers',
        'stövlar': 'boots',
        'sandaler': 'sandals',
        'väskor': 'bags handbags',
        'accessoarer': 'accessories',
        'bälten': 'belts',
        'halsdukar': 'scarves',
        'mössor': 'hats caps beanies',
        'handskar': 'gloves',
        'badkläder': 'swimwear bikinis',
        'träningskläder': 'activewear sportswear',
        'pyjamas': 'pajamas sleepwear',
        'kläder': 'clothing apparel',
        'herr': 'men mens male',
        'dam': 'women womens female',
        'barn': 'kids children',
        'man': 'men mens male man',
        'woman': 'women womens female woman',
    }

    # Gender term mappings (common variations → database values)
    GENDER_ALIASES = {
        'male': 'man men mens',
        'female': 'woman women womens',
        'men': 'man male mens',
        'women': 'woman female womens',
        'herr': 'man male men',
        'dam': 'woman female women',
        'unisex': 'unisex',
    }

    def __init__(self, config: Config):
        self.config = config
        self.model = SentenceTransformer(config.sentence_model)
        # Move to GPU for faster encoding
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

        self.local_colors = []
        self.local_categories = []
        self.local_genders = []

    def _add_english_aliases(self, swedish_text: str) -> str:
        """Add English translations to Swedish category text."""
        result = swedish_text
        lower_text = swedish_text.lower()
        for swedish, english in self.SWEDISH_TO_ENGLISH.items():
            if swedish in lower_text:
                result = f"{result} {english}"
        return result

    def set_local_colors(self, colors: list[dict]):
        self.local_colors = colors
        names = [c["name"] for c in colors]
        self.color_embeddings = self.model.encode(names, convert_to_tensor=True, show_progress_bar=False)

    def set_local_categories(self, categories: list[dict]):
        self.local_categories = categories
        texts = []
        for c in categories:
            text = c.get("path", c["name"]) or c["name"]
            if c.get("gender_name"):
                text = f"{c['gender_name']} {text}"
            # Add English translations for Swedish terms
            text = self._add_english_aliases(text)
            texts.append(text)
        logger.info(f"Category embedding examples: {texts[:3]}")
        self.category_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def set_local_genders(self, genders: list[dict]):
        self.local_genders = genders
        # Enrich gender names with common aliases
        texts = []
        for g in genders:
            name = g["name"]
            aliases = self.GENDER_ALIASES.get(name.lower(), '')
            text = f"{name} {aliases}".strip()
            texts.append(text)
        logger.info(f"Gender embeddings: {texts}")
        self.gender_embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

    def map_batch_colors(self, detected: list[str], csv_colors: list[str],
                          detected_confidences: list[float] = None) -> list[dict]:
        """Map batch of colors to local taxonomy."""
        if not self.local_colors:
            return [{"id": None, "name": None, "confidence": 0.0} for _ in detected]

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

        query_emb = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), self.color_embeddings.unsqueeze(0), dim=2)
        best_idx = sims.argmax(dim=1)
        best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        return [
            {"id": self.local_colors[idx.item()]["id"], "name": self.local_colors[idx.item()]["name"], "confidence": round(score.item(), 3)}
            for idx, score in zip(best_idx, best_scores)
        ]

    def map_batch_categories(self, detected: list[str], product_types: list[str],
                              google_categories: list[str] = None, detected_confidences: list[float] = None,
                              titles: list[str] = None) -> list[dict]:
        """
        Map batch of categories to local taxonomy.
        Priority: product_type > title keywords > detected from image (if confident) > google_category
        """
        if not self.local_categories:
            return [{"id": None, "name": None, "confidence": 0.0} for _ in detected]

        if google_categories is None:
            google_categories = [''] * len(detected)
        if detected_confidences is None:
            detected_confidences = [1.0] * len(detected)
        if titles is None:
            titles = [''] * len(detected)

        # Minimum confidence to use image detection
        MIN_DETECTION_CONFIDENCE = 0.15

        # Build query with priority: product_type > title > detected (if confident) > google
        queries = []
        for d, p, g, conf, title in zip(detected, product_types, google_categories, detected_confidences, titles):
            parts = []

            # 1. Product type is usually most specific (e.g., "Half zip sweater", "Jacket", "T-shirt")
            if p:
                parts.append(p)

            # 2. Title often contains product keywords (e.g., "M's Tree Message Tee" → "Tee")
            if title:
                # Extract last word which often indicates product type
                title_words = title.split()
                if title_words:
                    # Common product keywords at end of title
                    last_word = title_words[-1].rstrip(',').lower()
                    product_keywords = ['jacket', 'coat', 'tee', 't-shirt', 'shirt', 'pants', 'jeans',
                                       'shorts', 'dress', 'skirt', 'sweater', 'hoodie', 'top', 'blouse']
                    if last_word in product_keywords:
                        parts.append(last_word)

            # 3. Detected from image - ONLY if confidence is above threshold
            if d and conf >= MIN_DETECTION_CONFIDENCE:
                parts.append(d)

            # 4. Google category - only use last segment if specific enough
            if g and '>' in g:
                last_segment = g.split('>')[-1].strip()
                # Skip if too generic
                if last_segment and last_segment.lower() not in ['clothing', 'apparel', 'accessories', 'other']:
                    parts.append(last_segment)

            queries.append(' '.join(parts) if parts else 'unknown')

        query_emb = self.model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
        sims = torch.nn.functional.cosine_similarity(query_emb.unsqueeze(1), self.category_embeddings.unsqueeze(0), dim=2)
        best_idx = sims.argmax(dim=1)
        best_scores = sims.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        return [
            {"id": self.local_categories[idx.item()]["id"], "name": self.local_categories[idx.item()]["name"], "confidence": round(score.item(), 3)}
            for idx, score in zip(best_idx, best_scores)
        ]

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

        return [
            {"id": self.local_categories[idx.item()]["id"], "name": self.local_categories[idx.item()]["name"], "confidence": round(score.item(), 3)}
            for idx, score in zip(best_idx, best_scores)
        ]


# =============================================================================
# SIZE TYPE CLASSIFIER - Runs in parallel
# =============================================================================

class SizeTypeClassifier:
    """Zero-shot classifier for size type - uses all available context."""

    def __init__(self, config: Config):
        self.config = config
        self.device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "zero-shot-classification",
            model=config.zero_shot_model,
            device=self.device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

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

        # Process with optimal batching
        results_list = []
        for out in tqdm(
            self.classifier(valid_texts, self.config.size_type_labels, batch_size=32),
            total=len(valid_texts),
            desc="Classifying size types",
            leave=False
        ):
            results_list.append(out)

        # Map back
        results = [{"size_type": None, "confidence": 0.0} for _ in contexts]
        for i, valid_idx in enumerate(valid_indices):
            results[valid_idx] = {
                "size_type": results_list[i]["labels"][0],
                "confidence": round(results_list[i]["scores"][0], 3)
            }
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

        logger.info("Loading Sentence Transformer...")
        self.mapper = TaxonomyMapper(self.config)
        self.mapper.set_local_colors(colors)
        self.mapper.set_local_categories(categories)
        self.mapper.set_local_genders(genders)

        logger.info("Loading BART classifier...")
        self.size_classifier = SizeTypeClassifier(self.config)

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

    async def run(self, input_path: str, output_path: str):
        """Run the full pipeline with parallel execution."""
        start_time = time.time()

        try:
            await self.initialize()

            # Read input
            products = self.read_csv(input_path)
            total = len(products)
            batch_size = self.config.batch_size
            num_batches = (total + batch_size - 1) // batch_size

            logger.info("=" * 60)
            logger.info(f"PROCESSING {total} PRODUCTS")
            logger.info(f"Batch size: {batch_size}, Total batches: {num_batches}")
            logger.info("=" * 60)

            # Extract text data for parallel processing
            csv_colors = [p.get('color', '') for p in products]
            csv_genders = [p.get('gender', '') for p in products]
            product_types = [p.get('product_type', '') for p in products]
            google_categories = [p.get('google_product_category', '') for p in products]
            titles = [p.get('title', '') for p in products]
            descriptions = [p.get('description', '') for p in products]

            # Build rich context for size classification (uses ALL available signals)
            size_contexts = [
                {
                    'title': p.get('title', ''),
                    'description': p.get('description', ''),
                    'product_type': p.get('product_type', ''),
                    'google_product_category': p.get('google_product_category', ''),
                }
                for p in products
            ]

            # PARALLEL TASK 1: Size type classification (doesn't need images!)
            # Now uses: title + description + product_type + google_product_category
            logger.info("Starting size type classification (parallel)...")
            size_task = asyncio.create_task(
                asyncio.to_thread(self.size_classifier.classify_all, size_contexts)
            )

            # PARALLEL TASK 2: Process images in batches
            all_clip_results = []
            async with FastImageDownloader(self.config) as downloader:
                with tqdm(total=num_batches, desc="Processing batches", unit="batch") as pbar:
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, total)
                        batch_products = products[start_idx:end_idx]

                        # Process batch
                        clip_results = await self.process_images_batch(batch_products, downloader)
                        all_clip_results.extend(clip_results)

                        # Update progress
                        pbar.set_postfix({
                            "products": f"{end_idx}/{total}",
                            "downloads": f"{downloader.stats['success']}/{downloader.stats['success'] + downloader.stats['failed']}"
                        })
                        pbar.update(1)

                logger.info(f"Image downloads: {downloader.stats['success']} success, {downloader.stats['failed']} failed")

            # Wait for size classification to complete
            logger.info("Waiting for size classification to complete...")
            size_results = await size_task

            # Extract detected values with confidence scores
            detected_colors = [r["color"] for r in all_clip_results]
            detected_color_confidences = [r["color_conf"] for r in all_clip_results]
            detected_categories = [r["category"] for r in all_clip_results]
            detected_category_confidences = [r["category_conf"] for r in all_clip_results]

            # Map to local taxonomy (fast, batch processing)
            # Pass confidence so we can ignore low-confidence detections (< 0.15)
            logger.info("Mapping to local taxonomy...")
            mapped_colors = self.mapper.map_batch_colors(
                detected_colors, csv_colors, detected_color_confidences
            )
            mapped_categories = self.mapper.map_batch_categories(
                detected_categories, product_types, google_categories, detected_category_confidences, titles
            )
            mapped_genders = self.mapper.map_batch_genders(csv_genders)

            # BETA: NLP-based category classification (text only, no images)
            logger.info("Running NLP category classification (beta)...")
            nlp_categories = self.mapper.nlp_classify_categories(titles, descriptions, product_types)

            # Enrich products
            logger.info("Enriching products...")
            for i, product in enumerate(products):
                # Color
                product['detected_color'] = detected_colors[i]
                product['detected_color_confidence'] = all_clip_results[i]["color_conf"]
                product['mapped_local_color_id'] = mapped_colors[i]["id"]
                product['mapped_local_color'] = mapped_colors[i]["name"]
                product['color_mapping_confidence'] = mapped_colors[i]["confidence"]

                # Category
                product['detected_category'] = detected_categories[i]
                product['detected_category_confidence'] = all_clip_results[i]["category_conf"]
                product['mapped_local_category_id'] = mapped_categories[i]["id"]
                product['mapped_local_category'] = mapped_categories[i]["name"]
                product['category_mapping_confidence'] = mapped_categories[i]["confidence"]

                # Size type
                product['size_type'] = size_results[i]["size_type"]
                product['size_type_confidence'] = size_results[i]["confidence"]

                # Gender
                product['mapped_local_gender_id'] = mapped_genders[i]["id"]
                product['mapped_local_gender'] = mapped_genders[i]["name"]
                product['gender_mapping_confidence'] = mapped_genders[i]["confidence"]

                # BETA: NLP-based category (text only, no images)
                product['beta_nlp_category_id'] = nlp_categories[i]["id"]
                product['beta_nlp_category'] = nlp_categories[i]["name"]
                product['beta_nlp_category_confidence'] = nlp_categories[i]["confidence"]

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
