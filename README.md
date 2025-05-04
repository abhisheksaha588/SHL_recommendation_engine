# Assessment Recommendation System using RAG and Gemini

This project implements a semantic recommendation engine for SHL assessments using a Retrieval-Augmented Generation (RAG) pipeline with Gemini embeddings. The system processes a catalog of assessments and enables users to retrieve the most relevant assessments based on a query or requirement.

---

## 1. Data Collection

The assessment data was scraped from the SHL Product Catalog using requests and BeautifulSoup and scrape all 366 assements data:

**Source:** https://www.shl.com/solutions/products/product-catalog/

### Fields Extracted:

- `data-entity-id`
- `Assessment Name`
- `Relative URL`
- `Remote Testing`
- `Adaptive/IRT`
- `Test Type`
- `Assessment Length (seperate logic)`

**Example Raw Record:**

```
4038,C Programming (New),https://www.shl.com/solutions/products/product-catalog/view/c-programming-new/,Yes,No,Knowledge & Skills,10
```

---

## 2. Data Cleaning

The scraped data was processed using `pandas` to remove null values, correct data types, and retain relevant columns.

**Final Format (assessment.csv):**

```
data-entity-id,Assessment Name,Remote Testing,Adaptive/IRT,Test Type,Assessment Length
3827,.NET Framework 4.5,Yes,Yes,Knowledge & Skills,30.0
```

---

## 3. Embedding and Vector Store Creation

The cleaned data was used to build a semantic index for similarity-based retrieval.

### Process:

- Converted CSV to JSON format.
- Extracted relevant fields and converted them to a list of strings for embedding.
- Used Google Gemini (`google-generativeai`) for text embeddings.
- Stored the embeddings using FAISS for efficient nearest-neighbor search.

### Files Generated:

- `vector.index`: FAISS vector index
- `vector_texts.pkl`: Pickled list of original assessment texts

---

## 4. RAG Pipeline Development

A custom module (`rag_pipeline.py`) was developed to manage the RAG pipeline.

**Workflow:**

1. User query is embedded using Gemini.
2. FAISS is used to retrieve similar assessments.
3. Results are returned as top-10 matches with metadata.

---

## 5. Web Application

A simple Flask-based web interface i developed to enable interaction with the model.

### Features:

- Search interface for entering hiring requirements.
- Results displayed in a structured table format.
- REST API endpoint for integration with other systems.

---

## 6. API Endpoint

**URL:** `/api/search`  
**Method:** POST  
**Request Body:**

```json
{
  "query": "I am hiring a Flutter developer and need a 45-minute personality test"
}
```

**Response:**

```json
{
  "results": [
    "Android Development (New) | Type: Knowledge & Skills | Remote: Yes | Adaptive: No | length: 7.0"
  ]
}
```

---


## 8. Considerations

- The relative URLs (links) were excluded from embedding to reduce token length and improve performance.
- However, the original data with `data-entity-id` and links is retained and can be rejoined at runtime for display or retrieval.
- This allows for the integration of full product links in future enhancements without modifying the embedding pipeline.
