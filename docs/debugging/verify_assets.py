import sqlite3
import os
from pathlib import Path

def verify():
    db_path = "data/macpro.db"
    qdrant_path = "data/qdrant"
    
    print(f"Checking Database: {db_path}")
    if not os.path.exists(db_path):
        print("Database not found!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Documents
    cursor.execute("SELECT COUNT(*) FROM document")
    print(f"Total Documents: {cursor.fetchone()[0]}")
    
    # Assets
    print("\nAssets by type:")
    cursor.execute("SELECT asset_type, COUNT(*) FROM asset GROUP BY asset_type")
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]}")
        
    # Sample images
    print("\nSample Images:")
    cursor.execute("SELECT id, page_id, path_or_uri FROM asset WHERE asset_type = 'image' LIMIT 5")
    for row in cursor.fetchall():
        print(f"  - ID: {row[0]}, Page: {row[1]}, Path: {row[2]}")
        
    conn.close()
    
    print(f"\nQdrant Path: {qdrant_path}")
    if os.path.exists(qdrant_path):
        print("Qdrant directory exists.")
    else:
        print("Qdrant directory NOT found.")

if __name__ == "__main__":
    verify()
