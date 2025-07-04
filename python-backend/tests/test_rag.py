import pytest

# O módulo rag_store está no mesmo pacote python-backend
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rag_store


@pytest.mark.parametrize(
    "question,expected_snippet",
    [
        (
            "Qual é a carga horária do curso de Python Fundamental 1?",
            "20 horas",
        ),
        (
            "Qual é o prazo para conclusão?",
            "31 de julho de 2025",
        ),
    ],
)
def test_rag_answers(question: str, expected_snippet: str):
    """
    Garante que o RAG retorna chunks que contenham a informação solicitada.
    O índice e o JSONL já existem em data/, gerados previamente pelo demo.
    """
    results = rag_store.query_rag(question, k=3)
    assert results, "Nenhum resultado retornado pelo RAG"

    # Verifica se algum dos chunks retornados contém o trecho esperado
    assert any(
        expected_snippet.lower() in r["chunk"].lower() for r in results
    ), f'O trecho "{expected_snippet}" não foi encontrado nos resultados do RAG.'
