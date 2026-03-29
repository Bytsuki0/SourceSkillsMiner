"""
generate_synthetic.py  (updated)

Generates synthetic developer profiles for 11 programmer categories.
Now adds a 'category' column as the last column — the training label.

CSV schema (14 columns):
  name, top1lang, top2lang, top3lang, top4lang, top5lang,
        top1lib,  top2lib,  top3lib,  top4lib,  top5lib,  top6lib,  top7lib,
        category

Usage
──────
    python generate_synthetic.py
    python generate_synthetic.py --per-category 50 --seed 42
    python generate_synthetic.py --append features.csv
"""

import csv
import os
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple

Weight = float
Pool   = List[Tuple[str, Weight]]

@dataclass
class CategoryProfile:
    name:      str
    label:     str
    languages: Pool
    libraries: Pool


CATEGORIES: List[CategoryProfile] = [

    CategoryProfile(
        name='Frontend Development', label='frontend',
        languages=[
            ('TypeScript', 10), ('JavaScript', 9), ('CSS',  8), ('HTML', 8),
            ('SCSS', 5), ('Svelte', 4), ('Vue', 4), ('Less', 2), ('CoffeeScript', 1),
        ],
        libraries=[
            ('react', 10), ('vue', 7), ('svelte', 6), ('angular', 5), ('next', 8),
            ('nuxt', 4), ('tailwindcss', 7), ('vite', 6), ('webpack', 5),
            ('styled-components', 4), ('radix-ui', 5), ('shadcn-ui', 4),
            ('framer-motion', 3), ('zustand', 5), ('redux', 4),
            ('react-hook-form', 5), ('react-router', 5), ('axios', 4),
            ('tanstack-query', 4), ('lodash', 3), ('date-fns', 3),
            ('classnames', 4), ('lucide-react', 4), ('zod', 4),
        ],
    ),

    CategoryProfile(
        name='Backend Development', label='backend',
        languages=[
            ('Python', 10), ('Go', 9), ('Java', 8), ('Ruby', 7), ('PHP', 6),
            ('JavaScript', 5), ('TypeScript', 5), ('Rust', 4), ('C#', 4),
            ('Kotlin', 3), ('Scala', 2),
        ],
        libraries=[
            ('express', 8), ('fastapi', 9), ('django', 9), ('flask', 7),
            ('gin', 7), ('actix-web', 4), ('rails', 7), ('laravel', 6),
            ('spring', 5), ('sqlalchemy', 7), ('prisma', 6), ('typeorm', 5),
            ('sequelize', 4), ('redis', 6), ('celery', 5), ('jwt', 5),
            ('bcrypt', 4), ('dotenv', 4), ('logging', 5), ('os', 4),
            ('sys', 4), ('json', 4), ('datetime', 4), ('pydantic', 5),
        ],
    ),

    CategoryProfile(
        name='Fullstack Development', label='fullstack',
        languages=[
            ('TypeScript', 9), ('JavaScript', 9), ('Python', 7), ('Ruby', 6),
            ('HTML', 7), ('CSS', 6), ('Go', 4), ('PHP', 3),
        ],
        libraries=[
            ('react', 8), ('next', 9), ('express', 7), ('prisma', 6),
            ('tailwindcss', 7), ('django', 5), ('rails', 5), ('axios', 5),
            ('zustand', 4), ('typeorm', 4), ('graphql', 5), ('trpc', 4),
            ('zod', 5), ('jest', 4), ('supertest', 3), ('react-router', 4),
            ('shadcn-ui', 4), ('dotenv', 4),
        ],
    ),

    CategoryProfile(
        name='API Development', label='api',
        languages=[
            ('Go', 10), ('Python', 9), ('TypeScript', 8), ('JavaScript', 7),
            ('Java', 6), ('Rust', 5), ('Kotlin', 4), ('C#', 4),
        ],
        libraries=[
            ('fastapi', 10), ('gin', 9), ('express', 8), ('axum', 5),
            ('actix-web', 5), ('graphql', 7), ('grpc', 6), ('protobuf', 5),
            ('swagger', 6), ('pydantic', 7), ('jwt', 6), ('oauth2', 5),
            ('redis', 5), ('httpx', 4), ('requests', 5), ('context', 5),
            ('net/http', 5), ('strings', 4), ('os', 4), ('json', 5),
            ('logging', 4), ('testing', 4), ('fmt', 5),
        ],
    ),

    CategoryProfile(
        name='Web Development', label='webdev',
        languages=[
            ('JavaScript', 10), ('HTML', 9), ('CSS', 8), ('PHP', 7),
            ('TypeScript', 6), ('Python', 5), ('Ruby', 5), ('SCSS', 4), ('Vue', 4),
        ],
        libraries=[
            ('react', 8), ('vue', 7), ('angular', 5), ('jquery', 5),
            ('bootstrap', 5), ('tailwindcss', 7), ('express', 6),
            ('laravel', 6), ('django', 5), ('rails', 5), ('axios', 5),
            ('webpack', 5), ('vite', 5), ('sass', 4), ('path', 4),
            ('fs', 4), ('http', 4),
        ],
    ),

    CategoryProfile(
        name='Mobile Development', label='mobile',
        languages=[
            ('Swift', 10), ('Kotlin', 10), ('Dart', 8), ('Objective-C', 6),
            ('Java', 7), ('TypeScript', 5), ('JavaScript', 5), ('C#', 4),
        ],
        libraries=[
            ('SwiftUI', 10), ('UIKit', 9), ('Foundation', 8), ('flutter', 9),
            ('react-native', 8), ('Combine', 6), ('XCTest', 5),
            ('android.content', 7), ('android.view', 7), ('java.util', 6),
            ('HealthKit', 4), ('CoreData', 5), ('CoreLocation', 4),
            ('AVFoundation', 4), ('AppKit', 4), ('com.android', 5),
            ('androidx', 6), ('kotlinx.coroutines', 5),
            ('jetpack-compose', 6), ('retrofit', 5), ('glide', 4),
        ],
    ),

    CategoryProfile(
        name='Cloud DevOps', label='devops',
        languages=[
            ('Shell', 10), ('Python', 9), ('Go', 7), ('HCL', 7),
            ('Dockerfile', 6), ('Makefile', 5), ('JavaScript', 4),
            ('TypeScript', 4),
        ],
        libraries=[
            ('boto3', 9), ('os', 8), ('sys', 7), ('subprocess', 7),
            ('logging', 7), ('json', 6), ('kubernetes', 7), ('terraform', 6),
            ('ansible', 5), ('docker', 6),
            ('github.com/aws/aws-sdk-go', 6),
            ('google.golang.org/api', 4), ('fmt', 5), ('strings', 5),
            ('context', 6), ('testing', 5), ('time', 5), ('click', 5),
            ('argparse', 4), ('pyyaml', 5), ('fabric', 4), ('paramiko', 4),
        ],
    ),

    CategoryProfile(
        name='Database Projects', label='database',
        languages=[
            ('Python', 9), ('Java', 8), ('Ruby', 7), ('Go', 7),
            ('JavaScript', 6), ('TypeScript', 5), ('SQL', 8), ('PLpgSQL', 4),
            ('Kotlin', 4),
        ],
        libraries=[
            ('sqlalchemy', 10), ('prisma', 9), ('active_record', 8),
            ('mongoose', 7), ('typeorm', 7), ('sequelize', 6), ('redis', 7),
            ('psycopg2', 6), ('pg', 5), ('mysql2', 5), ('sqlite3', 5),
            ('alembic', 5), ('java.sql', 5), ('hibernate', 5),
            ('liquibase', 4), ('database/sql', 5), ('gorm', 6),
            ('knex', 4), ('faker', 4),
        ],
    ),

    CategoryProfile(
        name='Automation Scripting', label='automation',
        languages=[
            ('Python', 10), ('Shell', 10), ('Ruby', 7), ('Perl', 6),
            ('JavaScript', 5), ('PowerShell', 8), ('Makefile', 4), ('Go', 4),
        ],
        libraries=[
            ('os', 10), ('sys', 9), ('re', 8), ('subprocess', 8),
            ('pathlib', 6), ('glob', 5), ('shutil', 5), ('argparse', 6),
            ('click', 5), ('logging', 6), ('json', 5), ('csv', 5),
            ('selenium', 5), ('requests', 5), ('pytest', 5), ('capybara', 4),
            ('rake', 5), ('fabric', 4), ('schedule', 4), ('watchdog', 4),
            ('time', 5), ('threading', 4), ('multiprocessing', 4),
        ],
    ),

    CategoryProfile(
        name='Data Science', label='datascience',
        languages=[
            ('Python', 10), ('Jupyter Notebook', 9), ('R', 7), ('Julia', 4),
            ('MATLAB', 3), ('SQL', 5), ('JavaScript', 3), ('Scala', 3),
        ],
        libraries=[
            ('numpy', 10), ('pandas', 10), ('matplotlib', 9), ('scipy', 7),
            ('seaborn', 7), ('sklearn', 8), ('statsmodels', 6), ('plotly', 6),
            ('bokeh', 4), ('jupyterlab', 5), ('os', 5), ('sys', 5),
            ('json', 4), ('csv', 4), ('datetime', 5), ('sqlalchemy', 4),
            ('boto3', 3), ('dask', 4), ('polars', 4), ('pyarrow', 4), ('re', 4),
        ],
    ),

    CategoryProfile(
        name='Machine Learning', label='ml',
        languages=[
            ('Python', 10), ('Jupyter Notebook', 9), ('C++', 5),
            ('Julia', 4), ('CUDA', 3), ('R', 3), ('TypeScript', 3),
        ],
        libraries=[
            ('torch', 10), ('tensorflow', 9), ('keras', 7), ('sklearn', 9),
            ('transformers', 8), ('numpy', 10), ('pandas', 8),
            ('matplotlib', 7), ('scipy', 5), ('huggingface-hub', 6),
            ('datasets', 6), ('accelerate', 5), ('einops', 4),
            ('lightning', 5), ('onnx', 4), ('optuna', 4), ('mlflow', 4),
            ('wandb', 5), ('cv2', 5), ('PIL', 4), ('tqdm', 6),
            ('os', 5), ('json', 4), ('logging', 4),
        ],
    ),
]

# Maps label → full category name (used in the category column)
LABEL_TO_NAME = {cat.label: cat.name for cat in CATEGORIES}

HEADER = [
    'name',
    'top1lang', 'top2lang', 'top3lang', 'top4lang', 'top5lang',
    'top1lib',  'top2lib',  'top3lib',  'top4lib',  'top5lib',  'top6lib', 'top7lib',
    'category',   # ← training label, always last
]


def _weighted_sample(pool: Pool, k: int, rng: random.Random) -> List[str]:
    if not pool:
        return []
    k = min(k, len(pool))
    chosen, remaining = [], list(pool)
    for _ in range(k):
        if not remaining:
            break
        total  = sum(w for _, w in remaining)
        r      = rng.uniform(0, total)
        cumul  = 0.0
        for idx, (item, w) in enumerate(remaining):
            cumul += w
            if r <= cumul:
                chosen.append(item)
                remaining.pop(idx)
                break
    return chosen


def generate_profile(cat: CategoryProfile, index: int, rng: random.Random) -> dict:
    langs = _weighted_sample(cat.languages, rng.randint(3, 5), rng)
    libs  = _weighted_sample(cat.libraries, rng.randint(3, 7), rng)
    langs += [''] * (5 - len(langs))
    libs  += [''] * (7 - len(libs))
    return {
        'name':     f"{cat.label}_synth_{index:04d}",
        'top1lang': langs[0], 'top2lang': langs[1], 'top3lang': langs[2],
        'top4lang': langs[3], 'top5lang': langs[4],
        'top1lib':  libs[0],  'top2lib':  libs[1],  'top3lib':  libs[2],
        'top4lib':  libs[3],  'top5lib':  libs[4],  'top6lib':  libs[5],
        'top7lib':  libs[6],
        'category': cat.name,   # full human-readable label
    }


def generate_csv(output_path: str, per_category: int, seed: int, append_to: str = '') -> None:
    rng = random.Random(seed)

    existing_rows: List[dict] = []
    if append_to and os.path.exists(append_to):
        with open(append_to, 'r', encoding='utf-8', newline='') as f:
            existing_rows = list(csv.DictReader(f))
        print(f"Appending to {append_to}  ({len(existing_rows)} existing rows)")
        output_path = append_to

    total = per_category * len(CATEGORIES)
    print(f"\nGenerating {per_category} × {len(CATEGORIES)} categories = {total} rows")

    new_rows, counters = [], {cat.label: 0 for cat in CATEGORIES}
    for cat in CATEGORIES:
        for _ in range(per_category):
            counters[cat.label] += 1
            new_rows.append(generate_profile(cat, counters[cat.label], rng))
        print(f"  {cat.name:35s}  {per_category} profiles")

    all_rows = existing_rows + new_rows
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nTotal rows: {len(all_rows)}  (real={len(existing_rows)}, synthetic={len(new_rows)})")
    print(f"Output → {os.path.abspath(output_path)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',       '-o', default='synthetic.csv')
    p.add_argument('--per-category', type=int, default=5000)
    p.add_argument('--seed',         type=int, default=42)
    p.add_argument('--append',       default='')
    args = p.parse_args()
    generate_csv(args.output, args.per_category, args.seed, args.append)

if __name__ == '__main__':
    main()
