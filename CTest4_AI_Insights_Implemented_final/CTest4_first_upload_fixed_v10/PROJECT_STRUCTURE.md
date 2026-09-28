# Project Structure After Reorganization

## Overview
The project has been reorganized into `frontend` and `backend` folders for better separation of concerns.

## Directory Structure

```
.
├── frontend/
│   └── app.py                          # Streamlit UI application
│
├── backend/
│   ├── analytics_engine.py             # Analytics processing engine
│   ├── mongo_data_filter.py            # MongoDB data filtering
│   ├── mongo_to_chroma.py              # MongoDB to ChromaDB conversion utility
│   ├── requirements.txt                # Python dependencies
│   ├── CHROMADB_DATA.md                # ChromaDB documentation
│   ├── lock_requirements.bat           # Dependency locking script
│   ├── app/                            # Shared application utilities
│   │   ├── __init__.py
│   │   └── parsers/
│   │       ├── __init__.py
│   │       ├── allure_parser.py
│   │       ├── common.py
│   │       ├── extent_parser.py
│   │       ├── junit_parser.py
│   │       └── report_parser.py
│   ├── chroma_db/                      # ChromaDB vector store
│   ├── analytics_results/              # Analytics output directory
│   └── tests/                          # Test suite
│       ├── test_chromadb_failure_retains_mongodb.py
│       ├── test_chromadb_snapshot.py
│       ├── test_mongo_data_filter.py
│       └── test_namespace.py
│
├── .venv/                              # Python virtual environment
├── __pycache__/
├── README.md                           # Project README
├── run.bat                             # Main run script (updated)
├── setup_venv.bat                      # Virtual environment setup
└── .gitignore
```

## Key Changes Made

### 1. **Frontend Reorganization**
   - Moved `app.py` to `frontend/app.py`
   - Updated imports to reference backend modules with `backend.` prefix
   - Updated data paths to point to backend directories:
     - `DEFAULT_CHROMA_PATH = "./backend/chroma_db"`
     - `DEFAULT_OUTPUT_DIR = "./backend/analytics_results"`

### 2. **Backend Reorganization**
   - Moved all backend logic to `backend/` folder:
     - `analytics_engine.py`
     - `mongo_data_filter.py`
     - `mongo_to_chroma.py`
     - `app/` (parsers and utilities)
     - `chroma_db/` (vector store)
     - `analytics_results/` (output)
     - `tests/` (test suite)

### 3. **Import Updates**
   - **frontend/app.py**: Changed imports to use `backend.` prefix
     ```python
     from backend.mongo_data_filter import filter_mongodb_records
     from backend.analytics_engine import generate_analytics_after_chroma, ...
     ```
   
   - **backend/mongo_to_chroma.py**: Added sys.path for relative imports
   - **backend/tests/**: Updated paths to reference `frontend/app.py` and backend modules

### 4. **Script Updates**
   - **run.bat**: Updated to run `frontend/app.py`
     ```batch
     ".venv\Scripts\python.exe" -m streamlit run frontend/app.py
     ```

## How to Run

Execute the updated run script:
```batch
.\run.bat
```

This will:
1. Set up the virtual environment if needed
2. Start the Streamlit app from `frontend/app.py`
3. The app will automatically access backend modules and data directories

## UI Layout
✓ The UI layout remains exactly the same - all visual functionality is preserved.

## Functionality Preserved
✓ All imports work correctly with the new structure
✓ Data paths point to the correct backend directories
✓ Tests can be run from the backend folder or project root
✓ ChromaDB and MongoDB functionality unchanged
✓ All parsers remain accessible to the backend
