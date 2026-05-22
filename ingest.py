#!/usr/bin/env python3
"""
MacPro AI — CLI ingest script.

Usage:
    python ingest.py --folder data/input
    python ingest.py --file data/input/report.pdf
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


async def main():
    parser = argparse.ArgumentParser(description="MacPro AI — Ingest medical documents")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Folder containing medical files to ingest")
    group.add_argument("--file", help="Single file to ingest")
    parser.add_argument("--no-recursive", action="store_true", help="Don't recurse into subfolders")
    args = parser.parse_args()

    from src.utils.database import init_db, get_session_factory
    from src.ingestion.pipeline import IngestionPipeline

    await init_db()
    factory = get_session_factory()

    async with factory() as session:
        pipeline = IngestionPipeline(db_session=session)

        if args.folder:
            docs = await pipeline.ingest_folder(
                args.folder, recursive=not args.no_recursive
            )
            print(f"\n✓ Ingested {len(docs)} documents from {args.folder}")
            for doc in docs:
                print(f"  [{doc.status.value}] {doc.filename} — {doc.page_count} page(s)")
        else:
            doc = await pipeline.ingest_file(args.file)
            if doc:
                print(f"\n✓ [{doc.status.value}] {doc.filename} — {doc.page_count} page(s) — id={doc.id}")
            else:
                print(f"\n✗ Failed to ingest {args.file}")
                sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
