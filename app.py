"""
Meshtastic Signal Coverage Prediction App

This FastAPI application provides an endpoint to predict Meshtastic signal coverage
using the ITM (Irregular Terrain Model). The prediction takes into account
transmitter and receiver characteristics, geographical location, and regional LoRa settings.

Key components:
- PredictRequest: Input payload structure for the prediction.
- load_config: Loads configuration values from environment variables.
- /predict endpoint: Provides signal coverage prediction in GeoJSON format.

Requirements:
- SRTM (Shuttle Radar Topography Mission) data tiles for terrain elevation
- geoprop-py submodule
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from geoprop import Tiles, Itm, Point, Climate
from regions import meshtastic_regions
from typing import Literal
import logging
import geojson
import h3
import sys
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Default configuration
config = {
    "tile_dir": os.path.join(os.getcwd(), "my_srtm_data"),
    "h3_res": 8,
    "max_distance_km": 100.0,
}

# Debugging: Verify the SRTM tile directory
if not os.path.exists(config["tile_dir"]):
    raise ValueError(f"Directory {config['tile_dir']} does not exist.")
else:
    logging.info(f"Directory {config['tile_dir']} exists. Contents: {os.listdir(config['tile_dir'])}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://meshplanner.mpatrick.dev"],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


def load_config() -> dict:
    """
    Loads configuration values from environment variables.

    Raises:
        - Logs an error and exits if required environment variables are not set.
    """
    def get_env_var(var_name: str, convert_type=None, default=None):
        value = os.getenv(var_name, default)
        if value is None:
            logging.error(
                f"{var_name} is required and not set in the environment variables."
            )
            sys.exit(1)
        try:
            return convert_type(value) if convert_type else value
        except ValueError as e:
            logging.error(f"Invalid value for {var_name}: {e}")
            sys.exit(1)

    return {
        "tile_dir": get_env_var("tile_dir"),
        "h3_res": get_env_var("h3_res", int),
        "max_distance_km": get_env_var("max_distance_km", float),
    }


# Load configuration and initialize SRTM tiles
logging.info("Loading configuration...")
config = load_config()
logging.info(f"Configuration loaded: {config}")
logging.info(f"Loading SRTM tiles...")
tiles = Tiles(config["tile_dir"])
logging.info(f'SRTM tiles loaded from: {config["tile_dir"]}')

# Initialize ITM model
itm = Itm(tiles, climate=Climate.ContinentalTemperate)


class PredictRequest(BaseModel):
    """
    Expected input payload for the /predict endpoint.
    """
    lat: float = Field(..., ge=-90, le=90, description="Transmitter latitude")
    lon: float = Field(..., ge=-180, le=180, description="Transmitter longitude")
    txh: float = Field(1.0, gt=0, description="Transmitter height in meters")
    rxh: float = Field(1.0, gt=0, description="Receiver height in meters")
    tx_gain: float = Field(1.0, ge=0, description="Transmitter gain in dB")
    rx_gain: float = Field(1.0, ge=0, description="Receiver gain in dB")
    region: Literal[*meshtastic_regions.keys()] = Field(
        "US", description="Meshtastic LoRa region code"
    )
    resolution: int = Field(8, ge=7, le=12, description="Simulation H3 cell resolution")


@app.post("/predict")
async def predict(payload: PredictRequest) -> JSONResponse:
    """
    Predicts Meshtastic signal coverage using the ITM model.

    Returns:
        - GeoJSON FeatureCollection with predicted RSSI values.
    """
    logging.info(f"Received prediction request: {payload.dict()}")
    if payload.region not in meshtastic_regions.keys():
        logging.error(f"Region {payload.region} is not valid.")
        raise HTTPException(
            status_code=404, detail=f"Region '{payload.region}' not found"
        )

    start_time = time.time()
    try:
        center = Point(payload.lat, payload.lon, payload.txh)
        prediction_h3 = itm.coverage(
            center,
            config["h3_res"],
            meshtastic_regions[payload.region]["frequency"] * 1e6,
            config["max_distance_km"],
            payload.rxh,
            rx_threshold_db=None,
        )
    except ValueError as e:
        logging.error(f"Model calculation error: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"Error generating model prediction: {str(e)}"
        )

    duration = time.time() - start_time
    logging.info(f"ITM model calculation completed in {duration:.2f} seconds.")

    tx_power = meshtastic_regions[payload.region]["transmit_power"]

    features = []
    for row in prediction_h3:
        hex_boundary = h3.h3_to_geo_boundary(hex(row[0]), geo_json=True)
        loss_db = row[2]
        model_rssi = tx_power + payload.tx_gain + payload.rx_gain - loss_db
        features.append(
            geojson.Feature(
                geometry=geojson.Polygon([hex_boundary]),
                properties={"model_rssi": model_rssi},
            )
        )

    feature_collection = geojson.FeatureCollection(features)
    return JSONResponse(content=feature_collection)


@app.get("/")
def serve_frontend():
    """
    Serves the index.html file as the frontend for the app.
    """
    return FileResponse("index.html")