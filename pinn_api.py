from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import csv
from typing import Dict, Any
# Replace with your actual PINN model import
from pinn_model import run_pinn_model  # Your PINN model implementation

app = FastAPI()

@app.post("/predict")
async def predict(
    conductivity: float = Form(...),
    radius: float = Form(...),
    depth: float = Form(...),
    geometry_file: UploadFile = File(...),
    surface_temp_file: UploadFile = File(...)
) -> Dict[str, Any]:
    try:
        # Create temporary files to store the uploaded CSV files
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_geometry:
            geometry_content = await geometry_file.read()
            temp_geometry.write(geometry_content)
            temp_geometry_path = temp_geometry.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_surface_temp:
            surface_temp_content = await surface_temp_file.read()
            temp_surface_temp.write(surface_temp_content)
            temp_surface_temp_path = temp_surface_temp.name

        # Run the PINN model with the inputs
        # The output should be a dict with "depth" and "radius" as doubles
        output = run_pinn_model(
            conductivity=conductivity,
            radius=radius,
            depth=depth,
            geometry_file_path=temp_geometry_path,
            surface_temp_file_path=temp_surface_temp_path
        )

        # Validate the output
        if not isinstance(output, dict) or "depth" not in output or "radius" not in output:
            raise ValueError("Model output must be a dict with 'depth' and 'radius' keys")

        # Clean up temporary files
        os.remove(temp_geometry_path)
        os.remove(temp_surface_temp_path)

        # Return the model output as JSON
        return JSONResponse(content={"status": "success", "output": output})

    except Exception as e:
        # Clean up files in case of error
        if 'temp_geometry_path' in locals() and os.path.exists(temp_geometry_path):
            os.remove(temp_geometry_path)
        if 'temp_surface_temp' in locals() and os.path.exists(temp_surface_temp_path):
            os.remove(temp_surface_temp_path)
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )