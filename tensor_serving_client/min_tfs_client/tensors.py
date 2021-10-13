from .types import DataType
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import numpy as np
from typing import Iterable, AnyStr, Tuple


def coerce_to_bytes(text: AnyStr) -> bytes:
    if isinstance(text, str):
        return text.encode("utf-8")
    else:
        return text


def write_values_to_tensor_proto(
    tensor_proto: TensorProto, values: Iterable, dtype: DataType
) -> TensorProto:
    proto_field = getattr(tensor_proto, f"{dtype.proto_field_name}")
    if dtype.is_numeric:
        proto_field.extend(values)
    else:
        proto_field.extend([coerce_to_bytes(v) for v in values])
    tensor_proto.tensor_content = values.tobytes()
    return tensor_proto


def ndarray_to_tensor_proto(ndarray: np.ndarray) -> TensorProto:
    dtype = DataType(ndarray.dtype.type)
    proto = TensorProto(
        dtype=dtype.enum,
        tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in ndarray.shape]),
    )
    proto = write_values_to_tensor_proto(tensor_proto=proto, values=ndarray.ravel(), dtype=dtype)
    return proto


def extract_shape(tensor_proto: TensorProto) -> Tuple[int, ...]:
    return tuple((int(d.size) for d in tensor_proto.tensor_shape.dim))


def tensor_proto_to_ndarray(tensor_proto: TensorProto) -> np.ndarray:
    dtype = DataType(tensor_proto.dtype)
    shape = extract_shape(tensor_proto)
    if tensor_proto.tensor_content:
        return (np.frombuffer(tensor_proto.tensor_content,
                              dtype=dtype.numpy_dtype).copy().reshape(shape))
    proto_values = getattr(tensor_proto, dtype.proto_field_name)
    return np.array(proto_values, dtype=dtype.numpy_dtype).reshape(*shape)
