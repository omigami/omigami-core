from prefect import Flow
from spec2vec import SpectrumDocument

from omigami.spectra_matching.spec2vec.tasks import CreateDocuments
from omigami.spectra_matching.spec2vec.tasks import CreateDocumentsParameters
from omigami.spectra_matching.storage import FSDataGateway


def test_create_documents(
    mock_default_config, cleaned_spectra_paths, cleaned_spectra_chunks, tmpdir
):
    ion_mode = "positive"
    fs_dgw = FSDataGateway()

    documents_directory = tmpdir / "spec2vec/documents/positive/small/2_decimals"

    parameters = CreateDocumentsParameters(
        output_directory=str(documents_directory),
        ion_mode=ion_mode,
        n_decimals=2,
    )

    path = CreateDocuments(fs_dgw, parameters).run(cleaned_spectra_paths[0])

    assert path == f"{documents_directory}/chunk_0.pickle"
    documents = fs_dgw.read_from_file(path)
    assert isinstance(documents[0], SpectrumDocument)
    assert len(documents) == len(cleaned_spectra_chunks[0])


def test_create_documents_map(
    mock_default_config, cleaned_spectra_paths, cleaned_spectra_chunks, tmpdir
):
    ion_mode = "positive"
    fs_dgw = FSDataGateway()
    documents_directory = tmpdir / "spec2vec/documents/positive/small/2_decimals"

    parameters = CreateDocumentsParameters(
        output_directory=str(documents_directory),
        ion_mode=ion_mode,
        n_decimals=2,
    )
    with Flow("test-flow") as test_flow:
        t = CreateDocuments(fs_dgw, parameters).map(cleaned_spectra_paths)

    res = test_flow.run()
    data = res.result[t].result

    assert len(data) == len(cleaned_spectra_paths)
    assert res.is_successful()
