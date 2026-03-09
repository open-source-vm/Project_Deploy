"""
inference.py — Model loading and prediction pipeline for EV-BMS Dashboard v4.

Pipeline:
    Sensor Inputs (IR, QC, QD, Tavg, Tmax, ChargeTime)
    → Feature Scaling → GRU Raw SoH → Cycle-Calibrated SoH
    → Auto RL State [SoH, Temp, Cycle, Current]
    → Double DQN Charging Decision
"""

import os
import numpy as np
import joblib
import streamlit as st
import h5py
import json
import zipfile
import tempfile
from tensorflow.keras.models import load_model, model_from_json

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRU_SEQUENCE_LENGTH = 20
GRU_NUM_FEATURES = 6
CYCLE_LIFE = 800

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

ACTION_LABELS = {0: "Decrease Charging", 1: "Maintain Charging", 2: "Increase Charging"}


def load_model_compat(filepath):
    """Compatibility loader to handle 'batch_shape' in legacy configs."""
    try:
        return load_model(filepath, compile=False)
    except Exception as e:
        if "batch_shape" not in str(e):
            raise e

    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as z:
            with z.open('config.json') as f:
                config_dict = json.load(f)

            def fix_config(cf):
                if isinstance(cf, dict):
                    if 'batch_shape' in cf:
                        cf['batch_input_shape'] = cf.pop('batch_shape')
                    for k, v in cf.items():
                        fix_config(v)
                elif isinstance(cf, list):
                    for item in cf:
                        fix_config(item)

            fix_config(config_dict)

            from tensorflow.keras.utils import deserialize_keras_object
            model = deserialize_keras_object(config_dict)

            with tempfile.TemporaryDirectory() as tmpdir:
                weights_path = os.path.join(tmpdir, 'weights.h5')
                with open(weights_path, 'wb') as wf:
                    wf.write(z.read('model.weights.h5'))
                model.load_weights(weights_path)
            return model
    else:
        with h5py.File(filepath, 'r') as f:
            model_config = f.attrs.get('model_config')
            if model_config is None:
                raise ValueError("No model_config found in HDF5 file.")
            model_config_str = model_config.decode('utf-8') if isinstance(model_config, bytes) else model_config
            config_dict = json.loads(model_config_str)

            def fix_config(cf):
                if isinstance(cf, dict):
                    if 'batch_shape' in cf:
                        cf['batch_input_shape'] = cf.pop('batch_shape')
                    for k, v in cf.items():
                        fix_config(v)
                elif isinstance(cf, list):
                    for item in cf:
                        fix_config(item)

            fix_config(config_dict)
            model = model_from_json(json.dumps(config_dict))

        model.load_weights(filepath)
        return model

# ---------------------------------------------------------------------------
# Model Loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Load GRU, Double-DQN models and scalers."""
    gru_model = load_model_compat(os.path.join(MODEL_DIR, "gru_soh_model.keras"))
    dqn_model = load_model_compat(os.path.join(MODEL_DIR, "double_dqn_calibrated.keras"))
    scaler_X = joblib.load(os.path.join(MODEL_DIR, "gru_scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, "gru_scaler_y.pkl"))
    return gru_model, dqn_model, scaler_X, scaler_y


# ---------------------------------------------------------------------------
# GRU Sequence
# ---------------------------------------------------------------------------
def prepare_gru_sequence(scaler_X, ir, qc, qd, tavg, tmax, chargetime):
    """Scale inputs → tile to (1, 20, 6) sequence."""
    raw = np.array([[ir, qc, qd, tavg, tmax, chargetime]])
    scaled = scaler_X.transform(raw)
    sequence = np.tile(scaled, (GRU_SEQUENCE_LENGTH, 1))
    return np.expand_dims(sequence, axis=0)


# ---------------------------------------------------------------------------
# SoH Prediction (raw)
# ---------------------------------------------------------------------------
def predict_soh(gru_model, scaler_y, sequence):
    """GRU inference → inverse-scale → raw SoH float."""
    pred_scaled = gru_model.predict(sequence, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0])


# ---------------------------------------------------------------------------
# Auto-derive Cycle & Current from sensor inputs
# ---------------------------------------------------------------------------
def estimate_cycle(qd, qc, cycle_life=CYCLE_LIFE):
    """Estimate cycle: int((1 - QD/QC) * 800)."""
    if qc <= 0:
        return 0
    ratio = max(0.0, 1.0 - qd / qc)
    return int(ratio * cycle_life)


def estimate_current(qc, chargetime):
    """Estimate charging current: QC / (ChargeTime / 3600)."""
    if chargetime <= 0:
        return 0.0
    return round(qc / (chargetime / 3600.0), 2)


# ---------------------------------------------------------------------------
# SoH Calibration
# ---------------------------------------------------------------------------
def calibrate_soh(raw_soh, cycle, cycle_life=CYCLE_LIFE):
    """Calibrate raw SoH using cycle degradation.

    degradation = 1 - (cycle / cycle_life)
    SoH_final   = raw_soh * (0.7 + 0.3 * degradation)
    Clamped to [0.5, 1.0].
    """
    degradation = max(0.0, 1.0 - cycle / cycle_life)
    soh_final = raw_soh * (0.7 + 0.3 * degradation)
    return float(np.clip(soh_final, 0.5, 1.0))


# ---------------------------------------------------------------------------
# Battery Health Classification (4 levels)
# ---------------------------------------------------------------------------
def get_battery_health_label(soh):
    """Classify health from calibrated SoH."""
    if soh > 0.90:
        return "Healthy"
    elif soh > 0.80:
        return "Moderate"
    elif soh > 0.70:
        return "Degrading"
    else:
        return "Severely Degraded"


# ---------------------------------------------------------------------------
# RL State Construction
# ---------------------------------------------------------------------------
def construct_rl_state(soh, temperature, cycle, current):
    """Build [SoH, Temp, Cycle, Current] → shape (1, 4)."""
    return np.array([[soh, temperature, cycle, current]])


# ---------------------------------------------------------------------------
# RL Action Prediction
# ---------------------------------------------------------------------------
def predict_rl_action(dqn_model, state):
    """Double DQN → action int + Q-values array."""
    q_values = dqn_model.predict(state, verbose=0)
    action = int(np.argmax(q_values[0]))
    return action, q_values[0].astype(float)
