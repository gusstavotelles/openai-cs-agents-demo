import sys
import pathlib
import uvicorn

# Adiciona python-backend ao PYTHONPATH para importar api.py
backend_dir = pathlib.Path(__file__).parent
sys.path.append(str(backend_dir))

if __name__ == "__main__":
    uvicorn.run("api:app", reload=True, host="0.0.0.0", port=8000)