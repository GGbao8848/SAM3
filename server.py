from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from PIL import Image
import torch
import numpy as np
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import asyncio
import json
import cv2

app = FastAPI()

# Global model and processor
model = None
processor = None
stop_annotation = False

class Prompt(BaseModel):
    class_id: str
    name: str
    threshold: float = 0.5  # Confidence threshold, default 0.5

class AnnotationRequest(BaseModel):
    image_dir: str
    output_dir: Optional[str] = None
    output_format: str = "segmentation"  # "segmentation" (default) or "bbox"
    prompts: List[Prompt]

def mask_to_polygon(mask, img_width, img_height):
    """
    Convert binary mask to normalized polygon coordinates.
    Returns list of polygons (each is a list of normalized x, y).
    """
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Simplify contour
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 3:
            continue
            
        # Normalize coordinates
        poly_points = []
        for point in approx:
            x, y = point[0]
            poly_points.append(x / img_width)
            poly_points.append(y / img_height)
            
        polygons.append(poly_points)
        
    return polygons

@app.on_event("startup")
async def load_model():
    """Load SAM3 model on startup"""
    global model, processor
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    print("Model loaded successfully!")

@app.post("/stop")
async def stop_annotation_task():
    """Stop the current annotation task"""
    global stop_annotation
    stop_annotation = True
    return {"status": "stopping"}

@app.post("/annotate")
async def annotate_images(request: AnnotationRequest):
    """
    Annotate images in a directory based on text prompts.
    Creates a 'labels' folder next to image_dir and saves masks as TXT files (YOLO format).
    Returns progress updates as Server-Sent Events.
    """
    global stop_annotation
    stop_annotation = False
    
    image_dir = Path(request.image_dir)
    
    # Validate image directory
    if not image_dir.exists():
        raise HTTPException(status_code=400, detail=f"Image directory does not exist: {image_dir}")
    
    if not image_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {image_dir}")
    
    # Determine output directory
    if request.output_dir and request.output_dir.strip():
        labels_dir = Path(request.output_dir)
    else:
        # Default: create labels directory next to image_dir
        labels_dir = image_dir.parent / "labels"
    
    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving annotations to: {labels_dir}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Get all image files
    image_files = [f for f in image_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        raise HTTPException(status_code=400, detail=f"No images found in directory: {image_dir}")
    
    async def generate_progress():
        total_images = len(image_files)
        processed = 0
        
        for idx, image_path in enumerate(image_files):
            if stop_annotation:
                yield f"data: {json.dumps({'type': 'stopped', 'processed': processed, 'total': total_images})}\n\n"
                break
                
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                img_width, img_height = image.size
                
                # Set image in processor
                inference_state = processor.set_image(image)
                
                # Prepare TXT file for this image
                txt_filename = image_path.stem + ".txt"
                txt_path = labels_dir / txt_filename
                
                annotations = []
                
                # Process each prompt
                for prompt in request.prompts:
                    if stop_annotation:
                        break
                        
                    try:
                        # Run inference with text prompt
                        output = processor.set_text_prompt(
                            state=inference_state, 
                            prompt=prompt.name
                        )
                        
                        masks = output["masks"]
                        boxes = output["boxes"]
                        scores = output["scores"]
                        
                        # Filter by confidence threshold and save all detections above it
                        above_threshold = scores >= prompt.threshold
                        num_detections = above_threshold.sum().item()
                        
                        if num_detections > 0:
                            # Get all boxes and scores above threshold
                            filtered_boxes = boxes[above_threshold].cpu().numpy()
                            filtered_scores = scores[above_threshold].cpu().numpy()
                            
                            # Save each detection
                            for i, (box, score) in enumerate(zip(filtered_boxes, filtered_scores)):
                                if request.output_format == "bbox":
                                    x1, y1, x2, y2 = box
                                    x_center = ((x1 + x2) / 2) / img_width
                                    y_center = ((y1 + y2) / 2) / img_height
                                    width = (x2 - x1) / img_width
                                    height = (y2 - y1) / img_height
                                    
                                    # Add annotation line: class_id x_center y_center width height
                                    annotation_line = f"{prompt.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                                    annotations.append(annotation_line)
                                else:
                                    # Segmentation format
                                    # Get the corresponding mask
                                    # Note: output["masks"] shape is (N_queries, 1, H, W)
                                    # We need to find which mask corresponds to this box/score
                                    # Since we filtered by score, we can use the boolean mask 'above_threshold'
                                    
                                    # Get all masks above threshold
                                    filtered_masks = masks[above_threshold].cpu().numpy()
                                    
                                    # Current mask
                                    mask_array = filtered_masks[i][0] > 0
                                    
                                    # Convert to polygons
                                    polygons = mask_to_polygon(mask_array, img_width, img_height)
                                    
                                    for poly in polygons:
                                        # Format: class_id x1 y1 x2 y2 ...
                                        poly_str = " ".join([f"{coord:.6f}" for coord in poly])
                                        annotation_line = f"{prompt.class_id} {poly_str}"
                                        annotations.append(annotation_line)
                            
                            # Send progress update
                            yield f"data: {json.dumps({'type': 'success', 'image': image_path.name, 'prompt': prompt.name, 'class_id': prompt.class_id, 'count': num_detections, 'threshold': prompt.threshold, 'processed': processed, 'total': total_images})}\n\n"
                        else:
                            yield f"data: {json.dumps({'type': 'warning', 'image': image_path.name, 'prompt': prompt.name, 'message': f'No detections above threshold {prompt.threshold}', 'processed': processed, 'total': total_images})}\n\n"
                            
                    except Exception as e:
                        yield f"data: {json.dumps({'type': 'error', 'image': image_path.name, 'prompt': prompt.name, 'message': str(e), 'processed': processed, 'total': total_images})}\n\n"
                
                # Save annotations to TXT file (incremental save)
                if annotations:
                    with open(txt_path, 'w') as f:
                        f.write('\n'.join(annotations))
                    yield f"data: {json.dumps({'type': 'saved', 'image': image_path.name, 'file': txt_filename, 'processed': processed, 'total': total_images})}\n\n"
                
                processed += 1
                
                # Send progress update
                progress_percent = int((processed / total_images) * 100)
                yield f"data: {json.dumps({'type': 'progress', 'processed': processed, 'total': total_images, 'percent': progress_percent})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'image': image_path.name, 'message': f'Failed to process image: {str(e)}', 'processed': processed, 'total': total_images})}\n\n"
        
        # Send completion message
        if not stop_annotation:
            yield f"data: {json.dumps({'type': 'complete', 'processed': processed, 'total': total_images, 'labels_dir': str(labels_dir)})}\n\n"
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the frontend HTML"""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SAM3 Batch Annotation</title></head>
    <body>
        <h1>SAM3 Batch Annotation Tool</h1>
        <p>Frontend not found. Please create static/index.html</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8847)
