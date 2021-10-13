from typing import Any, Dict, Optional, Union

import grpc
import numpy as np

from tensorflow_serving.apis.classification_pb2 import ClassificationRequest, ClassificationResponse
from tensorflow_serving.apis.predict_pb2 import PredictRequest, PredictResponse
from tensorflow_serving.apis.regression_pb2 import RegressionRequest, RegressionResponse
from tensorflow_serving.apis.get_model_status_pb2 import (
    GetModelStatusRequest,
    GetModelStatusResponse,
)
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from tensorflow_serving.apis.model_service_pb2_grpc import ModelServiceStub
from tensorflow_serving.apis.get_model_metadata_pb2 import  (
    GetModelMetadataRequest,
    GetModelMetadataResponse
)
from .tensors import ndarray_to_tensor_proto

RequestTypes = Union[PredictRequest, ClassificationRequest, RegressionRequest]
ResponseTypes = Union[PredictResponse, ClassificationResponse, RegressionResponse]


class TensorServingClient:
    def __init__(self, host: str, port: int) -> None:
        self._host_address = f"{host}:{port}"
        self._channel = grpc.insecure_channel(self._host_address)

    def _make_inference_request(
        self,
        model_name: str,
        input_dict: Dict[str, np.ndarray],
        request_pb: RequestTypes,
        timeout: int,
        model_version: Optional[int],
    ) -> ResponseTypes:

        request = request_pb()
        request.model_spec.name = model_name
        stub = PredictionServiceStub(self._channel)

        if model_version is not None:
            request.model_spec.version.value = model_version

        for k, v in input_dict.items():
            request.inputs[k].CopyFrom(ndarray_to_tensor_proto(v))
        return stub.Predict(request, timeout)

    def predict_request(
        self,
        model_name: str,
        input_dict: Dict[str, np.ndarray],
        timeout: int = 60,
        model_version: Optional[int] = None,
    ) -> PredictResponse:
        request_params: Dict[str, Any] = {
            "model_name": model_name,
            "model_version": model_version,
            "input_dict": input_dict,
            "request_pb": PredictRequest,
            "timeout": timeout,
        }
        return self._make_inference_request(**request_params)

    def classification_request(
        self,
        model_name: str,
        input_dict: Dict[str, np.ndarray],
        timeout: int = 60,
        model_version: Optional[int] = None,
    ) -> ClassificationResponse:
        request_params: Dict[str, Any] = {
            "model_name": model_name,
            "model_version": model_version,
            "input_dict": input_dict,
            "request_pb": ClassificationRequest,
            "timeout": timeout,
        }
        return self._make_inference_request(**request_params)

    def regression_request(
        self,
        model_name: str,
        input_dict: Dict[str, np.ndarray],
        timeout: int = 60,
        model_version: Optional[int] = None,
    ) -> RegressionResponse:
        request_params: Dict[str, Any] = {
            "model_name": model_name,
            "model_version": model_version,
            "input_dict": input_dict,
            "request_pb": RegressionRequest,
            "timeout": timeout,
        }
        return self._make_inference_request(**request_params)

    # def model_status_request(
    #     self,
    #     model_name: str,
    #     model_version: Optional[int] = None,
    #     timeout: Optional[int] = 10,
    # ) -> GetModelStatusResponse:
    #     stub = ModelServiceStub(self._channel)
    #     request = GetModelStatusRequest()
    #     request.model_spec.name = model_name
    #     if model_version:
    #         request.model_spec.version.value = model_version
    #     result = stub.GetModelStatus(request, timeout)
    #     del stub
    #     return result

    def model_metadata_request(
        self,
        model_name: str,
        model_version: Optional[int] = None,
        timeout: Optional[int] = 10,
    ) -> GetModelMetadataResponse:
        # stub = PredictionServiceStub(self._channel)
        request = GetModelMetadataRequest()
        request.model_spec.name = model_name
        stub = PredictionServiceStub(self._channel)
        if model_version is not None:
            request.model_spec.version.value = model_version
        request.metadata_field.append("signature_def")
        return stub.GetModelMetadata(request, timeout)
