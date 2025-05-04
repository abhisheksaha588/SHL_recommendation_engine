# SHL Assessment Recommendation Engine with RAG and Gemini

This project showcases a semantic recommendation system for SHL assessments using a Retrieval-Augmented Generation (RAG) framework enhanced by Gemini embeddings. It enables users to query and discover the most relevant assessments based on their hiring or skill evaluation needs.

---
Try it out @ https://shl-recommendation-engine-d52x.onrender.com

## 1. Data Acquisition

Assessment data was programmatically scraped from SHL’s product catalog using requests and BeautifulSoup. A total of 366 assessments were extracted.

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

## 2. Data Preprocessing

The raw data was cleaned using pandas. Steps included handling missing values, enforcing consistent data types, and filtering the necessary columns.

**Final Format (assessment.csv):**

```
data-entity-id,Assessment Name,Remote Testing,Adaptive/IRT,Test Type,Assessment Length
3827,.NET Framework 4.5,Yes,Yes,Knowledge & Skills,30.0
```

---

## 3. Embedding Generation & Vector Indexing

Post-cleaning, the dataset was embedded for semantic search and indexed for efficient retrieval.

### Steps:

- Transformed the CSV into JSON format.
- Extracted key fields and concatenated them into string representations.
- Applied Gemini embeddings via google-generativeai.
- Stored the resulting vectors in a FAISS index for similarity-based search.

### Generated Artifacts:

- vector.index: FAISS index of embeddings
- vector_texts.pkl: Pickled list of original input strings

---

## 4. RAG-Based Search Pipeline

A dedicated module (rag_pipeline.py) manages the core Retrieval-Augmented Generation logic.

Pipeline Overview:

User query is embedded using Gemini.
Similar assessments are retrieved from the FAISS index.
The system returns the top 10 matches with relevant metadata

---

## 5. Interactive Web Interface

A lightweight Flask-based web app facilitates user interaction with the recommendation system.

### Core Features:

Search bar to enter hiring requirements.

Tabulated results of relevant assessments.

REST API support for Banckend integration.


---

## 6. REST API

**Endpoint: /api/search
**Method: POST
**Expected Request Format:

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


## 8. Design Notes & Future Considerations

-Embedding excluded URLs to minimize token count and improve efficiency.

-Original metadata (like data-entity-id and full links) is preserved and can be reintegrated during result rendering.

-This design supports future enhancements, such as displaying full product pages, without altering the core embedding logic
