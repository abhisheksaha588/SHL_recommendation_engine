# import os
# import pickle
# import logging
# from typing import List
# from pathlib import Path

# import pandas as pd
# import numpy as np
# import google.generativeai as genai

# from rag_recommender.modules.ingestion import load_assessments
# genai.configure(api_key=os.getenv("GENAI_API_KEY"))

# # Setup
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# EMBED_MODEL = "models/embedding-001"
# EMBEDDINGS_PATH = Path("embeddings.npy")
# TEXTS_PATH = Path("vector_texts.pkl")


# def generate_embedding(text: str) -> List[float]:
#     response = genai.embed_content(
#         model=EMBED_MODEL,
#         content=text,
#         task_type="retrieval_document"
#     )
#     return response["embedding"]

# def prepare_texts(df: pd.DataFrame) -> List[str]:
#     return df.apply(
#         lambda row: f"{row['Assessment Name']} | Type: {row['Test Type']} | "
#                     f"Remote: {row['Remote Testing']} | Adaptive: {row['Adaptive/IRT']} | length: {row['Assessment Length']}",
#         axis=1
#     ).tolist()


# if __name__ == "__main__":
#     df = load_assessments()
#     texts = prepare_texts(df)
#     embeddings = [generate_embedding(text) for text in texts]
#     embedding_matrix = np.array(embeddings).astype("float32")

#     np.save(EMBEDDINGS_PATH, embedding_matrix)
#     with open(TEXTS_PATH, "wb") as f:
#         pickle.dump(texts, f)

#     logging.info("Embeddings and texts saved successfully.")


import os
import pickle
import logging
from typing import List, Dict
from pathlib import Path

import pandas as pd
import numpy as np
import google.generativeai as genai

from rag_recommender.modules.ingestion import load_assessments

# Configure with ADC (no need for GENAI_API_KEY if using ADC)
genai.configure()

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

EMBED_MODEL = "models/embedding-001"
EMBEDDINGS_PATH = Path("embeddings.npy")
TEXTS_PATH = Path("vector_texts.pkl")


def generate_embedding(text: str) -> List[float]:
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]


def prepare_texts(df: pd.DataFrame) -> List[Dict[str, str]]:
    return df.apply(
        lambda row: {
            "text": f"{row['Assessment Name']} | Type: {row['Test Type']} | "
                    f"Remote: {row['Remote Testing']} | Adaptive: {row['Adaptive/IRT']} | "
                    f"length: {row['Assessment Length']}",
            "url": row["Catalog URL"]  # <- Ensure this column exists in your Excel/CSV
        },
        axis=1
    ).tolist()


if __name__ == "__main__":
    df = load_assessments()

    texts = prepare_texts(df)
    text_only = [item["text"] for item in texts]  # For embedding

    embeddings = [generate_embedding(text) for text in text_only]
    embedding_matrix = np.array(embeddings).astype("float32")

    np.save(EMBEDDINGS_PATH, embedding_matrix)
    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(texts, f)  # Save text + url

    logging.info("✅ Embeddings and texts with URLs saved successfully.")
