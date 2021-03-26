from spec2vec import SpectrumDocument
from spec2vec_mlops.tasks.convert_to_documents import DocumentConverter


def test_convert_to_documents(cleaned_data):
    converter = DocumentConverter()
    for spectrum in cleaned_data:
        document = converter.convert_to_document(spectrum, 1)
        assert isinstance(document, SpectrumDocument)
        assert document.words
