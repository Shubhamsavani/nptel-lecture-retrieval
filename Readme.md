# NPTEL Lecture Retrieval

An AI-powered lecture retrieval system for NPTEL courses that helps users efficiently search, retrieve, and interact with lecture content using semantic search and information retrieval techniques.

## Features

* Lecture retrieval based on user queries
* Semantic search over lecture content
* Fast and efficient indexing
* Course-wise lecture organization
* Easy-to-use interface
* Scalable retrieval pipeline

## Tech Stack

* Python
* Machine Learning / NLP
* Vector Search / Embeddings
* Streamlit / Flask (if applicable)
* FAISS / ChromaDB (if applicable)

## Project Structure

```bash
nptel-lecture-retrieval/
│
├── data/                # Dataset and processed lecture files
├── notebooks/           # Experiment and development notebooks
├── src/                 # Core source code
├── models/              # Saved models or embeddings
├── requirements.txt     # Python dependencies
├── app.py               # Main application entry point
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubhamsavani/nptel-lecture-retrieval.git
cd nptel-lecture-retrieval
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / MacOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python app.py
```

Or launch the notebook workflow:

```bash
jupyter notebook
```

## How It Works

1. Lecture content is collected and preprocessed.
2. Text embeddings are generated using NLP models.
3. Embeddings are indexed for efficient retrieval.
4. User queries are converted into embeddings.
5. Similar lectures/content are retrieved and displayed.

## Future Improvements

* Add multilingual support
* Improve retrieval accuracy using RAG pipelines
* Add lecture summarization
* Integrate voice-based search
* Deploy as a web application

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License.
