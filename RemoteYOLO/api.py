from dataclasses import dataclass
from typing import Literal
from blacksheep import post, status_code, FormPart, Application, Request
from blacksheep.server.headers.cache import cache_control
import io
import PIL.Image
from PIL.Image import Image
from PIL import ImageFilter, ImageEnhance, ImageDraw
from yolo import YOLOModelFormat, YOLOModel, YOLOAnalysisManager, AnalysisResult, test_model
from ultralytics.engine.results import Results
from datetime import datetime
import os
import shutil
from pathlib import Path
import uuid
import requests
import supervision as sv
import numpy as np
from trackers import DeepSORTFeatureExtractor, DeepSORTTracker
import base64
from time import perf_counter

debug = True


if debug:
    import logging
    log = logging.getLogger("uvicorn.error")
    log.setLevel(logging.DEBUG)

# USE THIS IF THE SERVICES ARE RUNNING INDEPENDENTLY
CLIP_ENDPOINT = "http://127.0.0.1:44778/api/clip"

# USE THIS IF USING THE DOCKER COMPOSE SCRIPT
# CLIP_ENDPOINT = "http://clip:44778/api/clip"
CROPS_ROOT = Path("/tmp/yolo-crops")

reid_model = DeepSORTFeatureExtractor.from_timm("mobilenetv4_conv_small.e1200_r224_in1k")
tracker = DeepSORTTracker(minimum_consecutive_frames=1, feature_extractor=reid_model, minimum_iou_threshold=0.2, appearance_threshold=0.75, appearance_weight=0.75)
annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
tracker.reset()

def iou_xyxy(a, b):
    # a: (M,4), b: (N,4)
    ax1, ay1, ax2, ay2 = a[:,0:1], a[:,1:2], a[:,2:3], a[:,3:4]
    bx1, by1, bx2, by2 = b[:,0], b[:,1], b[:,2], b[:,3]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-6))  # (M,N)

# keeps a track of the most confident guess of movie title for each tracker ID
tracker_movies = {}


def log_output_time(label: str, start_time: float):
    # log.debug(f"Output time | {label}={(perf_counter() - start_time) * 1000:.2f}")
    return

@dataclass
class YOLOAnalysisRequest:
    model: YOLOModel | Literal["undefined"]
    format: YOLOModelFormat | Literal["undefined"]
    image: Image
    
    def __init__(self, model: YOLOModel, format: YOLOModelFormat, image: list[FormPart]):
        self.model = model if str(model) in [i.lower() for i in YOLOModel._member_names_] else "undefined"
        self.format = format if str(format) in [i.lower() for i in YOLOModelFormat._member_names_] else "undefined"
        self.image = PIL.Image.open(io.BytesIO(image[0].data))

app = Application()

# clear all stored tracks and detected movies
@post ("/api/reset")
async def reset_trackers():
    tracker_movies.clear()
    tracker.reset()
    return status_code(200, {"success": True})

@post("/api/analyse")
@cache_control(no_cache=True, no_store=True)
async def analyse_image(request: Request):
    start = datetime.now()
    request_start = perf_counter()

    headers_start = perf_counter()
    content_type_header = request.headers.get_first(b"Content-Type")
    log_output_time("analyse_image.headers_ms", headers_start)
    
    if content_type_header == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    if content_type_header.decode().split(";")[0] != "multipart/form-data":
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    form_start = perf_counter()
    untyped_form = await request.form()
    log_output_time("analyse_image.form_parse_ms", form_start)
    
    if untyped_form == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    form: dict = untyped_form
    
    format = form.get("format", YOLOModelFormat.NCNN)
    model = form.get("model", YOLOModel.YOLO11N)
    
    if "image" not in form:
        return status_code(400, {"success": False, "error": "You must provide an image"})    
    
    if type(form["image"]) != list:
        return status_code(400, {"success": False, "error": "Provided image is not in the correct format"})
    
    request_build_start = perf_counter()
    analysis_request: YOLOAnalysisRequest = YOLOAnalysisRequest(model, format, form["image"])
    log_output_time("analyse_image.request_build_ms", request_build_start)
    
    if analysis_request.model == "undefined":
        return status_code(400, {"success": False, "error": "The provided YOLO model is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModel.__members__.keys())])})
        
    if analysis_request.format == "undefined":
        return status_code(400, {"success": False, "error": "The provided format is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModelFormat.__members__.keys())])})
    
    yolo_start = perf_counter()
    analysis_result: AnalysisResult = YOLOAnalysisManager.analyse_image(analysis_request.model, analysis_request.format, analysis_request.image)
    log_output_time("analyse_image.yolo_ms", yolo_start)

    if not analysis_result.success or analysis_result.result == None:
        return status_code(500, {"success": False, "error": analysis_result.reason})
    
    result: Results = analysis_result.result

    end = datetime.now()

    clip_response = None
    summary_start = perf_counter()
    summary = result.summary()
    log_output_time("analyse_image.summary_ms", summary_start)

    crop_prep_start = perf_counter()
    request_id = uuid.uuid4().hex
    req_dir = CROPS_ROOT / request_id
    req_dir.mkdir(parents=True, exist_ok=True)

    log_output_time("analyse_image.crop_prep_ms", crop_prep_start)

    crop_write_start = perf_counter()
    try:
        for i, det in enumerate(summary):
            xmin = int(det["box"].get("x1",0))
            ymin = int(det["box"].get("y1",0))
            xmax = int(det["box"].get("x2",0))
            ymax = int(det["box"].get("y2",0))

            crop = analysis_request.image.crop((xmin,ymin,xmax,ymax))
            crop_path = req_dir / f"crop{i}.jpg"
            crop.save(crop_path, format="JPEG")

        log_output_time("analyse_image.crop_write_ms", crop_write_start)

        clip_start = perf_counter()
        try:
            r = requests.post(CLIP_ENDPOINT, json={"images_path": str(req_dir)})
            if r.status_code == 200:
                clip_response = r.json()
            else:
                clip_response = {"error": f"CLIP service returned {r.status_code}"}
        except Exception as e:
            clip_response = {"error": str(e)}
        log_output_time("analyse_image.clip_request_ms", clip_start)
    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)


    postprocess_start = perf_counter()
    for i, entry in enumerate(summary):
        c = entry.pop("class")
        entry['class_id'] = c
        try:
            entry['movie_id'] = clip_response["movie_ids"][i] if clip_response else "null"
        except KeyError:
            entry['movie_id'] = clip_response["error"]
    log_output_time("analyse_image.postprocess_ms", postprocess_start)

    log_output_time("analyse_image.total_ms", request_start)
    response_data = {"success": True,  "metadata": {"speed": result.speed, "names": result.names, "request":{"time_ms": (end-start).microseconds/1000}}, "result":summary}
    return response_data   


@cache_control(no_cache=True, no_store=True)
async def analyse_image_sort(request: Request):
    start = datetime.now()
    request_start = perf_counter()

    headers_start = perf_counter()
    content_type_header = request.headers.get_first(b"Content-Type")
    log_output_time("analyse_image_sort.headers_ms", headers_start)
    
    if content_type_header == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    if content_type_header.decode().split(";")[0] != "multipart/form-data":
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    form_start = perf_counter()
    untyped_form = await request.form()
    log_output_time("analyse_image_sort.form_parse_ms", form_start)
    
    if untyped_form == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    form: dict = untyped_form
    
    format = form.get("format", YOLOModelFormat.NCNN)
    model = form.get("model", YOLOModel.YOLO11N)
    image_type = form.get("image_type", "blur")

    if "image" not in form:
        return status_code(400, {"success": False, "error": "You must provide an image"})    
    
    if type(form["image"]) != list:
        return status_code(400, {"success": False, "error": "Provided image is not in the correct format"})
    
    request_build_start = perf_counter()
    analysis_request: YOLOAnalysisRequest = YOLOAnalysisRequest(model, format, form["image"])
    log_output_time("analyse_image_sort.request_build_ms", request_build_start)
    
    if analysis_request.model == "undefined":
        return status_code(400, {"success": False, "error": "The provided YOLO model is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModel.__members__.keys())])})
        
    if analysis_request.format == "undefined":
        return status_code(400, {"success": False, "error": "The provided format is not supported. Please use one of " + ", ".join([f"'{i.lower()}'" for i in list(YOLOModelFormat.__members__.keys())])})
    
    yolo_start = perf_counter()
    analysis_result: AnalysisResult = YOLOAnalysisManager.analyse_image(analysis_request.model, analysis_request.format, analysis_request.image)
    log_output_time("analyse_image_sort.yolo_ms", yolo_start)

    if not analysis_result.success or analysis_result.result == None:
        return status_code(500, {"success": False, "error": analysis_result.reason})
    
    result: Results = analysis_result.result

    YOLO_end = datetime.now()

    CLIP_start = datetime.now()
    clip_response = None
    summary_start = perf_counter()
    summary = result.summary()
    log_output_time("analyse_image_sort.summary_ms", summary_start)

    crop_prep_start = perf_counter()
    request_id = uuid.uuid4().hex
    req_dir = CROPS_ROOT / request_id
    req_dir.mkdir(parents=True, exist_ok=True)

    crops_base64 = []
    log_output_time("analyse_image_sort.crop_prep_ms", crop_prep_start)

    # if debug:
    #     debug_overlay_start = perf_counter()
    #     yolo_boxes = np.array([[e["box"]["x1"], e["box"]["y1"], e["box"]["x2"], e["box"]["y2"]] for e in summary], dtype=np.float32)

    #     # Draw debugging markers on a copy so we never mutate the original request image.
    #     debug_image = analysis_request.image.copy().convert("RGB")
    #     img_w, img_h = debug_image.size
    #     # log.debug(f"YOLO detected\n{yolo_boxes}\nin image of size (width={img_w}, height={img_h})")

    #     # Draw 10 px thick red borders for each YOLO detection box, clamped to image bounds.
    #     draw = ImageDraw.Draw(debug_image)
    #     for box in yolo_boxes:
    #         x1, y1, x2, y2 = box.astype(int)
    #         x1 = max(0, min(x1, img_w - 1))
    #         x2 = max(0, min(x2, img_w - 1))
    #         y1 = max(0, min(y1, img_h - 1))
    #         y2 = max(0, min(y2, img_h - 1))
    #         left, right = sorted((x1, x2))
    #         top, bottom = sorted((y1, y2))
    #         # log.debug(f"Drawing red box border: left={left}, top={top}, right={right}, bottom={bottom}, thickness=10")
    #         draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=10)

    #     # Save the debug image to a temporary file for inspection
    #     debug_image_path = Path(f"debug/{datetime.now().timestamp() * 1000}_debug.jpg")
    #     debug_image_path.parent.mkdir(parents=True, exist_ok=True)
    #     debug_image.save(debug_image_path, format="JPEG")
    #     log_output_time("analyse_image_sort.debug_overlay_ms", debug_overlay_start)

    # save crops of YOLO detections to a temp folder to pass off to CLIP API
    crop_write_start = perf_counter()
    try:
        for i, det in enumerate(summary):
            xmin = int(det["box"].get("x1",0))
            ymin = int(det["box"].get("y1",0))
            xmax = int(det["box"].get("x2",0))
            ymax = int(det["box"].get("y2",0))

            crop = analysis_request.image.crop((xmin,ymin,xmax,ymax))
            crop_path = req_dir / f"crop{i:04d}.jpg"
            crop.save(crop_path, format="JPEG")
            standard_buffer = io.BytesIO()
            crop.save(standard_buffer, format="JPEG")

            blurred_buffer = io.BytesIO()
            blurred_crop = crop.filter(ImageFilter.GaussianBlur(radius=5))
            blurred_crop.save(blurred_buffer, format="JPEG")

            desat_buffer = io.BytesIO()
            desat_crop = crop.convert("L").convert("RGB")
            desat_crop.save(desat_buffer, format="JPEG")

            if (image_type == "blur"):
                crop_base64 = base64.b64encode(blurred_buffer.getvalue()).decode("utf-8")
            elif (image_type == "desaturated"):
                crop_base64 = base64.b64encode(desat_buffer.getvalue()).decode("utf-8")
            else:
                crop_base64 = None
            crops_base64.append(crop_base64)

        log_output_time("analyse_image_sort.crop_write_ms", crop_write_start)

        clip_start = perf_counter()
        try:
            r = requests.post(CLIP_ENDPOINT, json={"images_path": str(req_dir)})
            if r.status_code == 200:
                clip_response = r.json()
            else:
                clip_response = {"error": f"CLIP service returned {r.status_code}"}
        except Exception as e:
            clip_response = {"error": str(e)}
        log_output_time("analyse_image_sort.clip_request_ms", clip_start)
    finally:
        if req_dir.exists():
            shutil.rmtree(req_dir, ignore_errors=True)

    tracker_start = perf_counter()
    tracker_detections = sv.Detections.from_ultralytics(analysis_result.result)
    tracker_detections = tracker.update(tracker_detections, frame=np.array(analysis_request.image))
    log_output_time("analyse_image_sort.tracker_ms", tracker_start)

    # # Build a mapping from YOLO detection index to tracker ID
    # # by matching YOLO boxes to tracker boxes via IOU
    # yolo_boxes = np.array([[e["box"]["x1"], e["box"]["y1"], e["box"]["x2"], e["box"]["y2"]] for e in summary], dtype=np.float32)

    # trk_boxes = np.array(tracker_detections.xyxy, dtype=np.float32) if len(tracker_detections) > 0 else np.empty((0, 4), dtype=np.float32)
    # trk_ids = np.array(tracker_detections.tracker_id) if len(tracker_detections) > 0 else np.empty((0,), dtype=np.int64)

    # # Assign tracker IDs to YOLO detections using greedy IOU matching
    # assigned_ids = [None] * len(summary)
    # if len(trk_boxes) > 0 and len(yolo_boxes) > 0:
    #     ious = iou_xyxy(yolo_boxes, trk_boxes)  # (M, N)
        
    #     used_trackers = set()
    #     # Process in order of best possible match to avoid conflicts
    #     detection_order = sorted(range(len(summary)), key=lambda i: ious[i].max(), reverse=True)
        
    #     for i in detection_order:
    #         best_j = -1
    #         best_iou_val = 0.75  # Lowered threshold to catch more matches
    #         for j in range(len(trk_boxes)):
    #             if j not in used_trackers and ious[i, j] > best_iou_val:
    #                 best_iou_val = ious[i, j]
    #                 best_j = j
            
    #         if best_j >= 0:
    #             used_trackers.add(best_j)
    #             tid = trk_ids[best_j]
    #             try:
    #                 tid = int(tid)
    #             except Exception:
    #                 tid = None
    #             assigned_ids[i] = tid
    
    # PIL.Image.fromarray(annotator.annotate(np.array(analysis_request.image), tracker_detections, labels=tracker_detections.tracker_id)).save(f"{str(request_id)}.jpg")

    postprocess_start = perf_counter()
    for i, entry in enumerate(summary):
        c = entry.pop("class")
        entry["class_id"] = c
        entry["crop"] = crops_base64[i] if i < len(crops_base64) else None
        tracker_id = tracker_detections.tracker_id.tolist()[i]
        try:
            movie_details = clip_response["movie_ids"][i] if clip_response else "null"
            if movie_details == "null":
                continue
            if ((not tracker_id in tracker_movies) or tracker_movies[tracker_id]["confidence"] < movie_details["confidence"]):
                tracker_movies[tracker_id] = movie_details

            # entry["movie_id"] = tracker_movies[tracker_id]
            entry["movie_id"] = movie_details
        except KeyError:
            entry["movie_id"] = clip_response["error"]
        entry["track_id"] = tracker_id
    log_output_time("analyse_image_sort.postprocess_ms", postprocess_start)

    # clip_list = (clip_response or {}).get("movie_ids") or []
    
    # for i, entry in enumerate(summary):
    #     c = entry.pop("class")
    #     entry["class_id"] = c
    #     entry["crop"] = crops_base64[i] if i < len(crops_base64) else None

    #     tracker_id = assigned_ids[i]
    #     entry["track_id"] = tracker_id

    #     # Get CLIP prediction for THIS detection (indexed by i)
    #     clip_item = clip_list[i] if i < len(clip_list) else {"prediction": None, "confidence": 0.0}
    #     pred = clip_item.get("prediction")
    #     conf = float(clip_item.get("confidence") or 0.0)

    #     # If no tracker ID assigned, return current frame's prediction only
    #     if tracker_id is None:
    #         entry["movie_id"] = {"prediction": pred, "confidence": conf}
    #         continue

    #     # Get previous best guess for this tracker
    #     prev = tracker_movies.get(tracker_id)
        
    #     if prev is None:
    #         # First time seeing this track
    #         tracker_movies[tracker_id] = {"prediction": pred, "confidence": conf}
    #     elif pred is not None:
    #         prev_conf = float(prev.get("confidence") or 0.0)
    #         prev_pred = prev.get("prediction")
            
    #         # Update if:
    #         # 1. New prediction is more confident
    #         # 2. Previous had no prediction but now we have one
    #         # 3. Significantly more confident (>10% boost) even if different movie
    #         if (conf > prev_conf or 
    #             (prev_pred is None and pred is not None) or
    #             (conf > prev_conf + 0.1)):
    #             tracker_movies[tracker_id] = {"prediction": pred, "confidence": conf}

    #     entry["movie_id"] = tracker_movies[tracker_id].copy()

    CLIP_end = datetime.now()

    # log.debug(f"YOLO analysis time: {(YOLO_end-start).total_seconds()*1000:.2f} ms")
    # log.debug(f"CLIP analysis time: {(CLIP_end-CLIP_start).total_seconds()*1000:.2f} ms")
    # log.debug(f"Total request time: {(CLIP_end-start).total_seconds()*1000:.2f} ms")

    log_output_time("analyse_image_sort.total_ms", request_start)
    response_data = {"success": True,  "metadata": {"speed": result.speed, "names": result.names, "request":{"time_ms": (YOLO_end-start).total_seconds()*1000}}, "clip_time_ms":(CLIP_end-CLIP_start).total_seconds()*1000, "result":summary}
    return response_data

@post("/api/custom-model")
async def upload_custom_model(request: Request):
    start = datetime.now()
    request_start = perf_counter()
    
    headers_start = perf_counter()
    content_type_header = request.headers.get_first(b"Content-Type")        
    log_output_time("upload_custom_model.headers_ms", headers_start)
    
    if content_type_header == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data"})

    if content_type_header.decode().split(";")[0] != "multipart/form-data":
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    form_start = perf_counter()
    untyped_form = await request.form()
    log_output_time("upload_custom_model.form_parse_ms", form_start)
    
    if untyped_form == None:
        return status_code(405, {"success": False, "error": "This endpoint accepts only multipart/form-data, not " + content_type_header.decode().split(";")[0]})
    
    form: dict = untyped_form
    
    if "model" not in form:
        return status_code(400, {"success": False, "error": "You must provide a custom model."})    

    model = form.get("model")
    
    if model == None or type(model) != list:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    model_file: FormPart = model[0]

    if model_file == None:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    if model_file.file_name == None:
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    

    if not model_file.file_name.decode().endswith("pt"):
        return status_code(400, {"success": False, "error": "You must provide a valid custom model file in .pt format."})    
    
    file_bytes = model_file.data
    
    file_write_start = perf_counter()
    with open("custom_temp.pt", "xb") as f:
        f.write(file_bytes)
    log_output_time("upload_custom_model.file_write_ms", file_write_start)
    
    
    validate_start = perf_counter()
    valid = test_model("custom_temp.pt")
    log_output_time("upload_custom_model.validate_ms", validate_start)
    
    if not valid:
        os.remove("custom_temp.pt")
        return status_code(422, {"success": False, "error": "Provided custom model is not a valid YOLO model, please provide a valid YOLO model."})

    if os.path.exists("custom.pt") and os.path.isfile("custom.pt"):
        os.remove("custom.pt")
    if os.path.exists("custom_ncnn_model"):
        shutil.rmtree("custom_ncnn_model")
    if os.path.exists("custom.onnx") :
        os.remove("custom.onnx")
    os.rename("custom_temp.pt", "custom.pt")

    end = datetime.now()
    log_output_time("upload_custom_model.total_ms", request_start)

    return status_code(201, {"success": True, "result": "Accepted custom model " + model_file.file_name.decode(), "metadata":{"request":{"time_ms": (end-start).microseconds/1000}}})