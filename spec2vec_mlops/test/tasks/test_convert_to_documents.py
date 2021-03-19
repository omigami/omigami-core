from spec2vec import SpectrumDocument
from spec2vec_mlops.tasks.convert_to_documents import DocumentConverter


def test_convert_to_documents(cleaned_data):
    converter = DocumentConverter()

    documents = converter.convert_to_documents(cleaned_data, 1)
    assert len(documents) == len(cleaned_data)
    assert isinstance(documents[0], SpectrumDocument)
