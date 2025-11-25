"""
client.py

This script sends data to the deployed HTRU2 Pulsar Classification API.

-------------------------------------------------------------------------------

EXPECTED INPUT FORMAT
---------------------

The model expects 8 numerical features in this EXACT order:

    1. ip_mean      - Mean of the integrated profile
    2. ip_std       - Standard deviation of the integrated profile
    3. ip_kurtosis  - Excess kurtosis of the integrated profile
    4. ip_skewness  - Skewness of the integrated profile
    5. dm_mean      - Mean of the DM-SNR curve
    6. dm_std       - Standard deviation of the DM-SNR curve
    7. dm_kurtosis  - Excess kurtosis of the DM-SNR curve
    8. dm_skewness  - Skewness of the DM-SNR curve

-------------------------------------------------------------------------------

USAGE
-----

For single prediction:
    python client.py

For batch prediction:
    python client.py --batch

-------------------------------------------------------------------------------
"""

import requests
import json
import argparse


def single_prediction():
    """Send a single sample for prediction"""
    API_URL = "http://localhost:9696/predict"

    # Make sure features are in the exact order expected by the model
    payload = {
        "features": [
            99.3671875,  # ip_mean
            41.57220208,  # ip_std
            1.547196967,  # ip_kurtosis
            4.154106043,  # ip_skewness
            27.55518395,  # dm_mean
            61.71901588,  # dm_std
            2.20880796,  # dm_kurtosis
            3.662680136,  # dm_skewness
        ]
    }

    response = requests.post(API_URL, json=payload)

    print("=== SINGLE PREDICTION ===")
    print("Status:", response.status_code)
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.json())


def batch_prediction():
    """Send multiple samples for prediction"""
    API_URL = "http://localhost:9696/predict_batch"

    payload = {
        "samples": [
            [
                99.3671875,  # ip_mean
                41.57220208,  # ip_std
                1.547196967,  # ip_kurtosis
                4.154106043,  # ip_skewness
                27.55518395,  # dm_mean
                61.71901588,  # dm_std
                2.20880796,  # dm_kurtosis
                3.662680136,  # dm_skewness
            ],
            [
                140.0,  # ip_mean
                45.0,  # ip_std
                1.8,  # ip_kurtosis
                3.9,  # ip_skewness
                25.0,  # dm_mean
                60.0,  # dm_std
                2.1,  # dm_kurtosis
                3.5,  # dm_skewness
            ],
        ]
    }

    response = requests.post(API_URL, json=payload)

    print("=== BATCH PREDICTION ===")
    print("Status:", response.status_code)
    if response.status_code == 200:
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    else:
        print("Error:", response.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HTRU2 Pulsar Classification Client")
    parser.add_argument("--batch", action="store_true", help="Use batch prediction endpoint")

    args = parser.parse_args()

    if args.batch:
        batch_prediction()
    else:
        single_prediction()
