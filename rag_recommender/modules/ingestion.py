import pandas as pd
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "assessment.csv"

def load_assessments(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the assessments dataset without any preprocessing.
    """
    logging.info(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    logging.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
    return df

def convert_row_to_json(row) -> str:
    """
    Convert each row from the DataFrame to a structured JSON string for embedding.
    data-entity-id,Assessment Name,Relative URL,Remote Testing,Adaptive/IRT,Test Type,Assessment Length
    """

    assessment_json = {
        "data_entity_id": row["data-entity-id"],
        "assessment_name": row["Assessment Name"],
        "relative_url": row["Relative URL"],
        "remote_testing": row["Remote Testing"],
        "adaptive_irt": row["Adaptive/IRT"],
        "duration": row["Assessment Length"],  # Add the 'Duration' column if available
        "test_type": row["Test Type"]
    }
    return json.dumps(assessment_json)

def preprocess_and_convert_to_json(df: pd.DataFrame):
    """
    Convert all assessments from the DataFrame to JSON format.
    """
    logging.info("Converting each row into JSON format...")
    json_data = df.apply(convert_row_to_json, axis=1).tolist()
    return json_data

# Debug run
if __name__ == "__main__":
    df = load_assessments()
    json_data = preprocess_and_convert_to_json(df)
    print(json_data[:5])  # Print the first 5 JSON entries for debug
