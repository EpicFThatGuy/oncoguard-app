import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
from typing import List
import io
from PIL import Image

# Import local architecture modules
from dataset_streamer import ShardedMammographyStreamer
from model_architecture import DualBackboneOncoGuard, CalibratedInferenceEngine

app = FastAPI(title="OncoGuard Engine", description="Triage Engine for 500k+ Image Ingestion")

# Global Engine Loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_model = DualBackboneOncoGuard(pretrained=False)
# Load pre-trained weights if available
# raw_model.load_state_dict(torch.load("oncoguard_weights.pth", map_location=device))
raw_model.to(device)

engine = CalibratedInferenceEngine(raw_model, platt_a=1.12, platt_b=-0.45)
streamer = ShardedMammographyStreamer(shard_paths=[])

@app.post("/api/v1/triage_batch")
async def triage_batch(files: List[UploadFile] = File(...)):
    """
    Ingests batch uploads, executes parallel pre-processing, 
    calibrates probabilities, and outputs a sorted worklist.
    """
    worklist_results = []
    
    for file in files:
        try:
            contents = await file.read()
            pil_img = streamer.preprocess_clahe(contents)
            
            if pil_img is None:
                continue
                
            input_tensor = streamer.default_transform()(pil_img).unsqueeze(0).to(device)
            analysis = engine.predict(input_tensor)
            
            worklist_results.append({
                "filename": file.filename,
                "risk_score_pct": round(analysis["calibrated_probability"] * 100, 2),
                "priority": analysis["priority_code"],
                "target_sla": analysis["target_sla"]
            })
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            
    # Sort Worklist: Highest malignancy risk pushed directly to Index 0
    sorted_worklist = sorted(worklist_results, key=lambda x: x["risk_score_pct"], reverse=True)
    
    return {
        "processed_count": len(sorted_worklist),
        "top_priority_index_0": sorted_worklist[0] if sorted_worklist else None,
        "full_sorted_worklist": sorted_worklist
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
