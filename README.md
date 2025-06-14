# MarriageFlowTest

# Following is the directory structure: 


rag-app/
│
├── .env                          # Environment variables (API keys, etc.)
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview
│
├── config/
│   └── settings.py              # Configuration loader (loads .env, model settings)
│
├── data/
│   └── source_documents/        # Raw documents to index (PDFs, txts, etc.)
│
├── ingest/
│   ├── __init__.py
│   └── load_and_split.py        # Loads documents, chunks them
│   └── embed_and_store.py       # Embeds chunks & uploads to Pinecone
│
├── rag/
│   ├── __init__.py
│   └── retriever.py             # Wraps Pinecone retriever
│   └── generator.py             # Uses LangChain LLM + prompt templates
│   └── qa_chain.py              # Combines retriever + generator
│
├── app/
│   └── main.py                  # Streamlit or FastAPI app entry point
│   └── ui.py                    # Frontend components (if using Streamlit)
│
└── utils/
    └── logging.py               # Custom logger
    └── helpers.py               # Misc helpers
