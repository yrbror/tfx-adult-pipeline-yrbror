import tensorflow as tf
import tensorflow_transform as tft
from tensorflow import keras

# Import konfigurasi fitur dari preprocessing_clean
from preprocessing_clean import (
    NUMERIC_FEATURE_KEYS,
    CATEGORICAL_FEATURE_KEYS,
    LABEL_KEY,
    _xf,
)

# berapa bucket OOV untuk kategori
NUM_OOV_BUCKETS = 1


def _build_keras_model(tf_transform_output: tft.TFTransformOutput) -> keras.Model:
    """Bangun model Keras pakai fitur hasil Transform."""

    inputs = {}
    feature_list = []

    # ---- Numeric features: langsung jadi input dense ----
    for name in NUMERIC_FEATURE_KEYS:
        t_name = _xf(name)  # contoh: "age" -> "age_xf"
        inp = keras.Input(shape=(1,), name=t_name, dtype=tf.float32)
        inputs[t_name] = inp
        feature_list.append(inp)

    # ---- Categorical features: pakai embedding ----
    for name in CATEGORICAL_FEATURE_KEYS:
        t_name = _xf(name)
        # vocab pakai nama asli (bukan *_xf)
        vocab_size = tf_transform_output.vocabulary_size_by_name(name) + NUM_OOV_BUCKETS

        inp = keras.Input(shape=(1,), name=t_name, dtype=tf.int64)
        inputs[t_name] = inp

        # embedding
        emb = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=min(50, max(8, vocab_size // 2)),
        )(tf.cast(inp, tf.int32))
        emb = keras.layers.Reshape((-1,))(emb)
        feature_list.append(emb)

    # ---- Gabung semua fitur ----
    x = keras.layers.Concatenate()(feature_list)

    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )

    return model


def _input_fn(file_pattern, tf_transform_output, batch_size: int = 256):
    """Bikin tf.data.Dataset dari TFRecord hasil Transform.

    Catatan: TFRecord dari Transform biasanya GZIP → reader_args=["GZIP"]
    biar gak kena DataLossError.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        label_key=LABEL_KEY,  # label disimpan dengan nama asli, bukan *_xf
        reader=tf.data.TFRecordDataset,
        reader_args=["GZIP"],
    )

    return dataset


def run_fn(fn_args):
    """Entry point untuk TFX Trainer."""

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    # Debug dikit kalau mau lihat file yang dibaca
    print("=== TRAIN FILES PATTERN ===")
    print(fn_args.train_files)
    print("Sample train files:", tf.io.gfile.glob(fn_args.train_files)[:3])
    print("=== EVAL FILES PATTERN ===")
    print(fn_args.eval_files)
    print("Sample eval files :", tf.io.gfile.glob(fn_args.eval_files)[:3])

    train_dataset = _input_fn(
        fn_args.train_files,
        tf_transform_output,
        batch_size=256,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files,
        tf_transform_output,
        batch_size=256,
    )

    model = _build_keras_model(tf_transform_output)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
    )

    # Simpan model buat serving
    model.save(fn_args.serving_model_dir, save_format="tf")
    print("Model saved to:", fn_args.serving_model_dir)

