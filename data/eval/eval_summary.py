import json
import os
import csv

# ===================== CONFIG =====================
FILE_PREFIX = "results_E"
FILE_SUFFIX = ".json"
START_INDEX = 1
END_INDEX = 9

OUTPUT_CSV = "strategy_summary.csv"

# Custom strategy labels
STRATEGY_MAP = {
    "E1": "C1, no OCR, no BM25",
    "E2": "C1 + OCR",
    "E3": "C1 + OCR + BM25",
    "E4": "C2, no OCR, no BM25",
    "E5": "C2 + OCR",
    "E6": "C2 + OCR + BM25",
    "E7": "C3, no OCR, no BM25",
    "E8": "C3 + OCR",
    "E9": "C3 + OCR + BM25",
}

# Fields to keep
FIELDS = [
    "Strategy",
    "MRR",
    "Recall@5",
    "Recall@10",
    "LLM_judge",
    "annotated_n",
    "total_n",
    "MRR_conceptual"
]
# ==================================================

def process_files():
    processed_data = []

    for i in range(START_INDEX, END_INDEX + 1):
        filename = f"{FILE_PREFIX}{i}{FILE_SUFFIX}"

        if not os.path.exists(filename):
            print(f"Warning: {filename} not found, skipping.")
            continue

        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)

            exp_id = data.get("experiment")  # E1, E2, ...
            strategy_name = STRATEGY_MAP.get(exp_id, exp_id)

            row = {
                "Strategy": strategy_name,
                "MRR": data.get("MRR"),
                "Recall@5": data.get("Recall@5"),
                "Recall@10": data.get("Recall@10"),
                "LLM_judge": data.get("LLM_judge"),
                "annotated_n": data.get("annotated_n"),
                "total_n": data.get("total_n"),
                "MRR_conceptual": data.get("MRR_conceptual"),
            }

            processed_data.append(row)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return processed_data


def print_table(data):
    if not data:
        print("No data found.")
        return

    print("\n" + "-" * 100)
    print(" | ".join(FIELDS))
    print("-" * 100)

    for row in data:
        values = [str(row.get(col, "")) for col in FIELDS]
        print(" | ".join(values))

    print("-" * 100)


def save_csv(data):
    if not data:
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data)

    print(f"\nCSV saved as: {OUTPUT_CSV}")


def main():
    data = process_files()
    print_table(data)
    save_csv(data)


if __name__ == "__main__":
    main()