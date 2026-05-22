

```bash
docs/
└── debugging/
    ├── verify_assets.py          # Phase 2: SQLite asset counting
    ├── check_all_types.py        # Phase 3: Qdrant asset type enumeration
    ├── check_case.py             # Phase 3: Payload field case checking
    ├── check_qdrant_all.py       # Phase 3: Collection listing
    ├── debug_retrieval.py        # Phase 3: Raw text/image hit inspection
    ├── inspect_payload.py        # Phase 3: Qdrant payload structure
    ├── inspect_qdrant.py         # Phase 3: Image vector inspection
    ├── inspect_qdrant_v2.py      # Phase 3: Image vector deep inspection
    ├── inspect_clip_methods.py   # Phase 4: CLIP model method discovery
    ├── test_clip_load.py         # Phase 4: CLIP embedding test (reproduces crash)
    ├── test_clip_type.py         # Phase 4: Return type diagnosis
    └── test_clip_fix.py          # Phase 5: Verification of the fix

```

