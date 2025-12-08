# Product Data Enrichment Pipeline

## Problem Statement

We receive product feeds from suppliers in CSV format containing product data like title, description, images, colors, sizes, etc. However:

1. **Color inconsistency**: Suppliers use arbitrary color names ("Evening Haze", "True Black", "Geyser Gray") that don't match our local color taxonomy
2. **Category mismatch**: Supplier categories ("Jacket") need to be mapped to our hierarchical category system
3. **No image-based validation**: Supplier-provided color/category may be wrong or missing - we need to detect from images
4. **Size ambiguity**: A size like "XL" or "42" could refer to different garment types (shirts, pants, shoes)
5. **Gender mapping**: Supplier gender values need to map to our `product_gender` table

## Solution

An AI-powered pipeline that:

1. **Analyzes product images** using CLIP to detect actual colors and clothing types
2. **Maps supplier values** to our local taxonomy using semantic similarity
3. **Classifies size types** to understand what garment the size refers to
4. **Outputs enriched CSV** with all mappings and confidence scores

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT                                    │
│     CSV Product Feed (semicolon-delimited, ISO-8859-1)          │
│     Fields: title, description, product_type, image_link,       │
│             gender, age_group, color, material, size            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 1. DATABASE LOADER                               │
│                                                                  │
│   Fetches from PostgreSQL:                                      │
│   • color table (id, name, hex_code) - active only              │
│   • category table (id, name, path, gender_id) - with hierarchy │
│   • product_gender table (id, name)                             │
│                                                                  │
│   These become our "local taxonomy" for mapping                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. ASYNC IMAGE DOWNLOADER                        │
│                                                                  │
│   • Downloads product images from image_link URLs               │
│   • 100 concurrent connections (aiohttp + semaphore)            │
│   • Handles failures gracefully (returns None for failed)       │
│   • Streams batches to GPU while downloading next batch         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. CLIP IMAGE ANALYSIS                           │
│                 Model: openai/clip-vit-base-patch32             │
│                                                                  │
│   For each product image, detects:                              │
│                                                                  │
│   COLOR DETECTION                  CATEGORY DETECTION            │
│   ┌─────────────────────┐         ┌─────────────────────┐       │
│   │ Scores image against│         │ Scores image against│       │
│   │ color prompts:      │         │ category prompts:   │       │
│   │ "red colored        │         │ "a photo of a       │       │
│   │  clothing"          │         │  jacket"            │       │
│   │ "blue colored       │         │ "a photo of jeans"  │       │
│   │  clothing"          │         │ "a photo of a       │       │
│   │ "black colored      │         │  t-shirt"           │       │
│   │  clothing"          │         │ etc...              │       │
│   │ etc...              │         │                     │       │
│   └─────────────────────┘         └─────────────────────┘       │
│                                                                  │
│   Output: detected_color, detected_category + confidence        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 4. TAXONOMY MAPPER                               │
│                 Model: sentence-transformers/all-MiniLM-L6-v2   │
│                                                                  │
│   Maps detected values + CSV values → local database IDs        │
│                                                                  │
│   COLOR MAPPING:                                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Input: detected_color ("black") + csv_color ("True Black")│  │
│   │ Process: Embed query, find nearest local color           │   │
│   │ Output: mapped_local_color_id, mapped_local_color        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   CATEGORY MAPPING:                                             │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Input: detected_category ("jacket") + product_type       │   │
│   │ Process: Embed query, find nearest local category        │   │
│   │          (uses category path + gender for context)       │   │
│   │ Output: mapped_local_category_id, mapped_local_category  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   GENDER MAPPING:                                               │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Input: csv_gender ("male", "Herr", "Men")               │   │
│   │ Process: Embed query, find nearest product_gender        │   │
│   │ Output: mapped_local_gender_id, mapped_local_gender      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 5. SIZE TYPE CLASSIFIER                          │
│                 Model: facebook/bart-large-mnli                  │
│                                                                  │
│   Determines what type of garment the size field refers to      │
│                                                                  │
│   Input: title + description                                    │
│   Labels: ["jeans/pants", "shirt/top", "dress", "shoes",       │
│            "socks", "underwear", "jacket/coat", "accessories",  │
│            "swimwear", "one-size"]                              │
│                                                                  │
│   Example:                                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Title: "Houdini M's Fall in Jacket, Evening Haze, XXL"  │   │
│   │ Size: "XXL"                                              │   │
│   │ → size_type: "jacket/coat" (confidence: 0.89)           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   This tells us the "XXL" is a jacket size, not pants/shoes    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                   │
│     Enriched CSV (same format: semicolon, ISO-8859-1)           │
│                                                                  │
│     Original columns preserved + new columns:                   │
│     • detected_color, detected_color_confidence                 │
│     • mapped_local_color_id, mapped_local_color                 │
│     • color_mapping_confidence                                  │
│     • detected_category, detected_category_confidence           │
│     • mapped_local_category_id, mapped_local_category           │
│     • category_mapping_confidence                               │
│     • size_type, size_type_confidence                           │
│     • mapped_local_gender_id, mapped_local_gender               │
│     • gender_mapping_confidence                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models Used

| Task | Model | Why This Model | VRAM |
|------|-------|----------------|------|
| Image → Color/Category | `openai/clip-vit-base-patch32` | Fast, good zero-shot, ~500 img/s | ~1GB |
| Text → Local Taxonomy | `sentence-transformers/all-MiniLM-L6-v2` | Fast semantic similarity | ~500MB |
| Size Type Classification | `facebook/bart-large-mnli` | Best zero-shot text classifier | ~1.5GB |

**Total VRAM**: ~3GB (fits easily on A10 24GB)

---

## Database Tables Used

From `schema.prisma`:

### color
```sql
SELECT id, name, hex_code FROM color
WHERE active = true AND deleted_at IS NULL
```

### category
```sql
SELECT c.id, c.name, c.parent_id, c.level, c.path, c.slug,
       c.gender_id, g.name as gender_name
FROM category c
LEFT JOIN product_gender g ON c.gender_id = g.id
WHERE c.active = true AND c.deleted_at IS NULL
```

### product_gender
```sql
SELECT id, name FROM product_gender
WHERE active = true AND deleted_at IS NULL
```

---

## Output Columns Explained

| Column | Source | Description |
|--------|--------|-------------|
| `detected_color` | CLIP (image) | Color detected from product image |
| `detected_color_confidence` | CLIP | How confident (0-1) |
| `mapped_local_color_id` | Sentence Transformer | ID in `color` table |
| `mapped_local_color` | Sentence Transformer | Matched color name |
| `color_mapping_confidence` | Sentence Transformer | Mapping confidence (0-1) |
| `detected_category` | CLIP (image) | Clothing type from image |
| `detected_category_confidence` | CLIP | How confident (0-1) |
| `mapped_local_category_id` | Sentence Transformer | ID in `category` table |
| `mapped_local_category` | Sentence Transformer | Matched category name |
| `category_mapping_confidence` | Sentence Transformer | Mapping confidence (0-1) |
| `size_type` | BART-MNLI | What garment the size is for |
| `size_type_confidence` | BART-MNLI | How confident (0-1) |
| `mapped_local_gender_id` | Sentence Transformer | ID in `product_gender` table |
| `mapped_local_gender` | Sentence Transformer | Matched gender name |
| `gender_mapping_confidence` | Sentence Transformer | Mapping confidence (0-1) |

---

## Performance

### Hardware: NVIDIA A10 (24GB VRAM)

| Stage | Time for 40K products |
|-------|----------------------|
| Database fetch | ~1 sec |
| Image download (async) | ~15-25 min |
| CLIP inference (batched) | ~2 min |
| Taxonomy mapping | ~30 sec |
| Size classification | ~7 min |
| **Total** | **~25-35 min** |

### Optimizations Applied

1. **Async I/O**: 100 concurrent image downloads
2. **GPU Batching**: Batch size 128 for CLIP
3. **FP16 Inference**: Half precision for 2x speedup
4. **Pre-computed Embeddings**: Text prompts embedded once
5. **Streaming Pipeline**: Download while processing

### Cost Estimate

At $0.75/hr (Lambda Labs A10): **~$0.35-0.45** per 40K products

---

## Project Structure

```
ai-test/
├── app.py              # Single-file pipeline (~800 lines)
├── requirements.txt    # Python dependencies
├── .env                # DATABASE_URL connection string
├── .env.example        # Template for .env
├── PLAN.md             # This file
├── inputs/
│   └── products.csv    # Input product feed
└── outputs/
    └── enriched_products.csv  # Output (auto-generated)
```

---

## Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure .env has DATABASE_URL
cat .env
# DATABASE_URL=postgresql://user:pass@host:5432/db

# 3. Place product CSV in inputs/
ls inputs/
# products.csv

# 4. Run pipeline
python app.py

# 5. Check output
ls outputs/
# enriched_products.csv
```

---

## Example Input → Output

### Input Row
```csv
"id";"title";"color";"size";"gender";"product_type";"image_link"
"820008_B77-XXL";"Houdini M's Fall in Jacket, Evening Haze, XXL";"Evening Haze";"XXL";"male";"Jacket";"https://..."
```

### Output Row (additional columns)
```csv
...original columns...
"detected_color";"black"
"detected_color_confidence";"0.72"
"mapped_local_color_id";"15"
"mapped_local_color";"Gray"
"color_mapping_confidence";"0.85"
"detected_category";"jacket"
"detected_category_confidence";"0.91"
"mapped_local_category_id";"234"
"mapped_local_category";"Jackets"
"category_mapping_confidence";"0.94"
"size_type";"jacket/coat"
"size_type_confidence";"0.89"
"mapped_local_gender_id";"1"
"mapped_local_gender";"Men"
"gender_mapping_confidence";"0.97"
```

---

## Future Improvements

1. **Use existing mapping tables**: Check `ie_color_mapping`, `ie_category_mapping` before AI inference
2. **Batch insert to mapping tables**: Auto-populate mapping tables with AI suggestions
3. **Confidence thresholds**: Flag low-confidence mappings for human review
4. **Image caching**: Cache downloaded images to SSD for re-runs
5. **Incremental processing**: Only process new/changed products
