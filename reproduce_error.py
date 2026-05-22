import asyncio
import os
from pathlib import Path

async def main():
    # Diagnostic: Check the file content BEFORE importing
    schema_path = Path("src/models/schema.py").resolve()
    print(f"--- DIAGNOSTIC: CONTENT OF {schema_path} ---")
    if schema_path.exists():
        with open(schema_path, "r") as f:
            lines = f.readlines()
            # Just print the relevant lines to avoid truncation
            for i, line in enumerate(lines):
                if "pages:" in line or "assets:" in line or "Mapped" in line:
                    print(f"{i+1}: {line.strip()}")
    else:
        print("FILE NOT FOUND!")
    print("--- END DIAGNOSTIC ---")

    try:
        from src.models.schema import Document, Page, Asset, ProcessingStatus, FileType
        print("Import successful!")
        doc = Document(
            filename="test.pdf",
            file_type=FileType.PDF,
            file_path="/tmp/test.pdf"
        )
        print("Successfully created Document instance")
    except Exception as e:
        print(f"Caught exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
