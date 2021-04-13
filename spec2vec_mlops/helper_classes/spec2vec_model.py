import flask
from flask import jsonify
from mlflow.pyfunc import PythonModel

from spec2vec_mlops import config

KEYS = config["gnps_json"]["necessary_keys"]
#
# """
# User Defined Exception
# """
# class UserCustomException(Exception):
#
#     status_code = 400
#
#     def __init__(self, message, http_status_code):
#         Exception.__init__(self)
#         self.message = message
#         self.status_code = http_status_code
#
#
# class Model(PythonModel):
#     pass
    # model_error_handler = flask.Blueprint("error_handlers", __name__)

#     @model_error_handler.app_errorhandler(UserCustomException)
#     def handleCustomError(error):
#         response = jsonify(error.to_dict())
#         response.status_code = error.status_code
#         return response
#
#     def __init__(
#         self,
#         model: Word2Vec,
#         n_decimals: int,
#         intensity_weighting_power: Union[float, int],
#         allowed_missing_percentage: Union[float, int],
#     ):
#         self.model = model
#         self.n_decimals = n_decimals
#         self.intensity_weighting_power = intensity_weighting_power
#         self.allowed_missing_percentage = allowed_missing_percentage
#         self.data_loader = DataLoader()
#         self.data_cleaner = DataCleaner()
#         self.document_converter = DocumentConverter()
#         self.embedding_maker = EmbeddingMaker(self.n_decimals)
#
#
#     def predict(self, context, model_input: List[Dict]) -> List[Dict]:
#         raise UserCustomException('Test-Error-Msg',400)
#         self._validate_input(model_input)
#         embeddings = self._pre_process_data(model_input)
#         # get library embeddings from feast
#         # for now going to use the calculated ones
#         best_matches = self._get_best_matches(embeddings, embeddings)
#         return best_matches
#
#     def _pre_process_data(self, model_input: List[Dict]) -> List[Embedding]:
#         cleaned_data = [self.data_cleaner.clean_data(data) for data in model_input]
#         documents = [
#             self.document_converter.convert_to_document(spectrum, self.n_decimals)
#             for spectrum in cleaned_data
#         ]
#         embeddings = [
#             self.embedding_maker.make_embedding(
#                 self.model,
#                 document,
#                 self.intensity_weighting_power,
#                 self.allowed_missing_percentage,
#             )
#             for document in documents
#         ]
#         return embeddings
#
#     def _get_best_matches(
#         self, references: List[Embedding], queries: List[Embedding]
#     ) -> List[Dict]:
#         spec2vec_embeddings_similarity = Spec2VecEmbeddings(
#             model=self.model,
#             intensity_weighting_power=self.intensity_weighting_power,
#             allowed_missing_percentage=self.allowed_missing_percentage,
#         )
#         scores = calculate_scores(
#             references,
#             queries,
#             spec2vec_embeddings_similarity,
#         )
#         best_matches = []
#         for i, query in enumerate(queries):
#             best_match = scores.scores_by_query(query, sort=True)[0]
#             best_matches.append(
#                 {
#                     "spectrum_number": i,
#                     "best_match_id": best_match[0].spectrum_id,
#                     "score": best_match[1],
#                 }
#             )
#         return best_matches
#
#     @staticmethod
#     def _validate_input(model_input: List[Dict]):
#         for spectrum in model_input:
#             if not isinstance(spectrum, Dict):
#                 raise ValidateInputException("Input data must be a dictionary", 1402, 402)
#
#             mandatory_keys = ["peaks_json", "Precursor_MZ"]
#             if any(key not in spectrum.keys() for key in mandatory_keys):
#                 raise MandatoryKeyMissingError(
#                     f"Please include all the mandatory keys in your input data. "
#                     f"The mandatory keys are {mandatory_keys}",
#                     1,
#                     400,
#                 )
#
#             if isinstance(spectrum["peaks_json"], str):
#                 try:
#                     ast.literal_eval(spectrum["peaks_json"])
#                 except ValueError:
#                     raise IncorrectPeaksJsonTypeError(
#                         "peaks_json needs to be a string representation of a list or a list",
#                         1,
#                         400,
#                     )
#             elif not isinstance(spectrum["peaks_json"], list):
#                 raise IncorrectPeaksJsonTypeError(
#                     "peaks_json needs to be a string representation of a list or a list",
#                     1,
#                     400,
#                 )
#
#             float_keys = ["Precursor_MZ", "Charge"]
#             for key in float_keys:
#                 if spectrum.get(key):
#                     try:
#                         float(spectrum[key])
#                     except ValueError:
#                         raise IncorrectFloatFieldTypeError(
#                             f"{key} needs to be a string representation of a float",
#                             1,
#                             400,
#                         )
#
#             for key in KEYS:
#                 if key not in float_keys + mandatory_keys:
#                     if not isinstance(spectrum.get(key, ""), str):
#                         raise IncorrectStringFieldTypeError(
#                             f"{key} needs to be a string", 1, 400
#                         )
#



"""
User Defined Exception
"""
class UserCustomException(Exception):

    status_code = 404

    def __init__(self, message, application_error_code,http_status_code):
        Exception.__init__(self)
        self.message = message
        if http_status_code is not None:
            self.status_code = http_status_code
        self.application_error_code = application_error_code

    def to_dict(self):
        rv = {"status": {"status": self.status_code, "message": self.message,
                         "app_code": self.application_error_code}}
        return rv

"""
Model Template
"""
class Model(PythonModel):

    """
    The field is used to register custom exceptions
    """
    model_error_handler = flask.Blueprint('error_handlers', __name__)

    """
    Register the handler for an exception
    """
    @model_error_handler.app_errorhandler(UserCustomException)
    def handleCustomError(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    def __init__(self, **kwargs):
        pass

    def predict(self, X, features_names, **kwargs):
        raise UserCustomException('Test-Error-Msg',1402,402)
        return X