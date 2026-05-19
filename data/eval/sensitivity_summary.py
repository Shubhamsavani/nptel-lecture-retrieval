import json
import os
import csv

# ===================== CONFIG =====================
FILE_PREFIX = "sensitivity_"
FILE_SUFFIX = ".json"

OUTPUT_CSV = "sensitivity_summary.csv"

FIELDS = [
    "Strategy",
    "MRR",
    "Recall@5",
    "Recall@10",
    "LLM_judge",
    "annotated_n",
    "total_n",
]
# ==================================================


def process_files():
    data = []

    for filename in os.listdir():
        # Match files like: sensitivity_C*.json
        if not (filename.startswith(FILE_PREFIX + "C") and filename.endswith(FILE_SUFFIX)):
            continue

        try:
            with open(filename, "r", encoding="utf-8") as f:
                content = json.load(f)

            # Remove prefix + suffix → keep full strategy
            strategy_name = filename[len(FILE_PREFIX):-len(FILE_SUFFIX)]
            # Examples:
            # sensitivity_C2_150.json -> C2_150
            # sensitivity_C3_0_25.json -> C3_0_25

            row = {
                "Strategy": strategy_name,
                "MRR": content.get("MRR"),
                "Recall@5": content.get("Recall@5"),
                "Recall@10": content.get("Recall@10"),
                "LLM_judge": content.get("LLM_judge"),
                "annotated_n": content.get("annotated_n"),
                "total_n": content.get("total_n"),
            }

            data.append(row)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Optional: sort naturally by strategy name
    data.sort(key=lambda x: x["Strategy"])

    return data


def print_table(data):
    if not data:
        print("No matching files found.")
        return

    print("\n" + "-" * 90)
    print(" | ".join(FIELDS))
    print("-" * 90)

    for row in data:
        values = [str(row.get(col, "")) for col in FIELDS]
        print(" | ".join(values))

    print("-" * 90)


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