
# 1. First Understand: What This File Is

This file is:

> Core Data Model of MacPro AI

It defines **how every piece of information in the system is structured, stored, and connected.**

Think of it as:

> The brain structure of the entire RAG system

Before writing ingestion, embeddings, OCR, or vector DB, we must define:

- what is a document
- what is a page
- what is an asset
- how they connect
- what data flows in the system

This file defines that.

---

## Layer 1 — What is this file doing?

Ask:

> What is the purpose of this file in the whole system?

For this file:

Answer:
This file defines **data models and database structure** for storing medical documents and extracted assets.

---

## Layer 2 — Why this file exists?

Ask:

> Why do we need this file in MacPro AI?

Answer:

Because RAG systems need structured storage of:

- documents
- pages
- images
- tables
- OCR
- X-rays
- metadata

Without structure, retrieval becomes impossible.

So this file ensures:

> Every piece of extracted data has identity and relationship.

---

## Layer 3 — Why this approach?

Ask:

> Why SQLModel?
> Why enums?
> Why Document → Page → Asset hierarchy?

This is where deep understanding happens.

--- 

## Layer 4 — Engineering reasoning

Ask:
> What problem does this design solve?

That gives real knowledge.



# Part 1
```
"""
MacPro AI — Unified data models.
Everything that flows through the pipeline is typed here.
SQLModel doubles as both Pydantic models AND SQLAlchemy ORM tables.
"""
```

##### What this means

This file defines **all data structures** used in MacPro AI.

#### Why SQLModel?

SQLModel combines:

**Pydantic**
Validation

and

**SQLAlchemy**
Database ORM

So instead of writing:
```
Pydantic model
SQLAlchemy model
Schema
```

You write only one.

---

##### Why this is smart

Because:

Medical RAG system has huge data.

We need:
Validation
Database
API response models
Serialization
All in one.

So SQLModel reduces complexity.

Part 2
class AssetType(str, Enum):
Why Enum?

Because we want controlled data types.

Instead of:

text
txt
Text
TEXT
image
img

We define:

TEXT
TABLE
IMAGE
OCR
URL
DICOM
Why this is important

In RAG:

Filtering happens.

Example:

Retrieve only images.

Retrieve only tables.

Retrieve only X-rays.

Enum ensures:

Consistency

Filtering

Clean metadata

Part 3
class FileType(str, Enum):

Why?

Because document types vary.

PDF

Image

DICOM

Unknown

This helps ingestion pipeline decide:

How to process file.

Example:

PDF → PyMuPDF

Image → OCR

DICOM → pydicom

So:

FileType controls pipeline flow.

Part 4
class ProcessingStatus(str, Enum):

Why needed?

Because ingestion is asynchronous.

File goes through stages.

Pending

Processing

Done

Failed

This helps:

Monitoring

Retry

Debugging

Tracking

Without this:

System becomes blind.

Part 5
Document Table
class Document(SQLModel, table=True):

This is top-level entity.

Why Document?

Because every file is a document.

PDF

X-ray

Report

Scan

DICOM

So we store:

filename

path

file type

page count

status

metadata

Why store metadata?

Medical documents have:

Patient

Study

Modality

Date

Description

This helps queries like:

Show Rahul X-ray

Show MRI study

Show CT scan

So document metadata supports filtering.

Part 6
pages: list["Page"]
assets: list["Asset"]

This defines relationships.

Why relationships?

Because:

Document contains pages

Pages contain assets

Assets contain images/tables/text

So structure becomes:

Document

→ Page

→ Asset

This is called:

Hierarchical data modeling.

Why hierarchical?

Because medical PDFs are hierarchical.

Document

Page

Content

So we match real-world structure.

Part 7
Page Table
class Page(SQLModel, table=True):

Why page?

Because retrieval happens at page level.

Not whole document.

Example:

Diagnosis is on page 5.

X-ray is on page 7.

Table is on page 3.

So page-level indexing is necessary.

Why raw_text?

To store extracted text.

Why image_path?

To store rendered page image.

Useful for:

Highlighting

OCR

UI rendering

Part 8
Asset Table

This is the most important.

class Asset(SQLModel, table=True):
Why Asset?

Because document contains many content types.

Text

Table

Image

OCR

URL

DICOM

Instead of making:

TextTable

ImageTable

URLTable

OCRTable

We create:

One unified asset table.

Why this is powerful

Because:

All content becomes searchable.

Unified structure.

Simplifies vector indexing.

Asset Fields
document_id

Link to document

page_id

Link to page

This creates:

Traceability

content

Stores:

Text

OCR text

Table

URL

path_or_uri

Stores:

Image path

X-ray

DICOM

Files

This allows:

Returning actual image.

bbox

Stores:

Position of asset on page.

Why?

For UI highlighting.

Example:

Highlight X-ray in page.

vector_id

Stores vector DB ID.

Why?

Because vector DB stores embeddings separately.

So we link.

Asset → Vector

This connects:

Database

Vector DB

LLM

meta

Extra information.

Example:

OCR confidence

Table headers

Image resolution

Medical info

Flexible storage.

Part 9
SourceReference

This is API response.

Why?

Because RAG must return evidence.

Not just answer.

So we send:

document

page

asset

snippet

image

score

This makes system explainable.

Important for medical.

Part 10
QueryResponse

API output.

Why structured?

Because frontend needs:

Answer

Sources

Query

Time

Not raw text.

Part 11
IngestRequest

Used for API.

User uploads folder.

System processes.

Part 12
IndexStats

Used for monitoring.

Documents

Pages

Assets

Vectors

Helps system health.

4. Why This Architecture Is Used

This design follows:

Document Graph Model

Document

→ Page

→ Asset

→ Vector

This solves:

Multimodal retrieval

Traceability

Scalability

Evidence linking

Medical compliance

5. Why Not Simpler Design?

Bad design would be:

One table with text.

That fails because:

No image linking

No page mapping

No OCR linking

No DICOM support

No vector linking

So Claude Code chose:

Structured multimodal schema.

6. How You Should Study Each File

Use this method.

File Study Template

For every file ask:

1 What is this file?
2 Why does it exist?
3 What problem does it solve?
4 Why this design?
5 How it connects to pipeline?
6 What would break if removed?

Write answers in notes.

7. Example Note You Should Write

models.py

Purpose:

Defines database and data structure.

Why:

To store documents, pages, and assets.

Problem solved:

Multimodal data linking.

Design:

Document → Page → Asset hierarchy.

Impact:

Enables vector retrieval and evidence linking.

This builds deep understanding.

8. Recommended Study Order

Study files in this order:

1 models.py

Data structure

↓

2 ingestion

How data enters

↓

3 parsers

How data extracted

↓

4 embeddings

How data converted to vectors

↓

5 indexing

How stored

↓

6 retrieval

How searched

↓

7 API

How user interacts

This matches pipeline.

9. Engineering Mindset

When Claude Code generates files, think:

This is not code.

This is system design.

Every file answers:

How data flows.

How knowledge is stored.

How retrieval happens.

How LLM answers.

