import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, Text, Any

# Kolom label
LABEL_KEY = "income"

# Kolom numerik persis seperti di adult.csv
NUMERIC_FEATURE_KEYS = [
    "age",
    "fnlwgt",
    "educational-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

# Kolom kategorikal persis seperti di adult.csv
CATEGORICAL_FEATURE_KEYS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "gender",
    "native-country",
]

def _xf(name: str) -> str:
    """Nama feature setelah transform."""
    return name + "_xf"


# ===== helper buat imputasi & casting =====

def _dense_float(x: tf.Tensor) -> tf.Tensor:
    """Pastikan tensor float dan isi nilai NaN dengan 0."""
    x = tf.cast(x, tf.float32)
    # ganti NaN (kalau ada) dengan 0
    x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
    return x


def _dense_string(x: tf.Tensor) -> tf.Tensor:
    """Pastikan tensor string dan isi string kosong dengan 'unknown'."""
    x = tf.cast(x, tf.string)
    # string kosong -> "unknown"
    x = tf.where(
        tf.equal(x, ""),
        tf.constant("unknown", dtype=tf.string),
        x,
    )
    return x


# ===== preprocessing utama yang dipakai TFX =====

def preprocessing_fn(inputs: Dict[Text, Any]) -> Dict[Text, Any]:
    """Preprocessing untuk seluruh fitur Adult dataset."""
    outputs: Dict[Text, Any] = {}

    # 1) Fitur numerik: impute + z-score
    for name in NUMERIC_FEATURE_KEYS:
        value = _dense_float(inputs[name])
        value = tft.scale_to_z_score(value)
        outputs[_xf(name)] = value

    # 2) Fitur kategorikal: impute + vocab -> ID
    for name in CATEGORICAL_FEATURE_KEYS:
        value = _dense_string(inputs[name])
        ids = tft.compute_and_apply_vocabulary(
            value,
            num_oov_buckets=1,
            vocab_filename=name,  # penting: sama dengan nama feature
        )
        outputs[_xf(name)] = ids

    # 3) Label: ">50K" -> 1, lainnya -> 0
    label = _dense_string(inputs[LABEL_KEY])
    label_int = tf.cast(tf.equal(label, ">50K"), tf.int64)
    outputs[LABEL_KEY] = label_int

    return outputs