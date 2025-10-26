import json
from pathlib import Path
import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Dict
from tqdm import tqdm

OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

COLORS = {
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (255, 0, 255),
    6: (0, 255, 255),
}

ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    filename = Path(image_path).name
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    # Load and validate image
    try:
        pil_image = Image.open(image_path)
        pil_image.verify()  # Verify it's a valid image
        pil_image = Image.open(image_path)  # Re-open after verify closes the file
    except (FileNotFoundError, IOError, Image.UnidentifiedImageError) as e:
        raise ValueError(f"Could not load image at {image_path}: {e}")
    
    img_width, img_height = pil_image.size
    draw = ImageDraw.Draw(pil_image)
    
    # Load detection data
    try:
        with open(info_path, 'r') as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load detection data from {info_path}: {e}")
        return np.array(pil_image)
    
    # Extract view index from filename
    _, view_index = extract_frame_info(image_path)
    
    # Get detections for this view
    try:
        detections = info["detections"]
        if view_index >= len(detections):
            print(f"Warning: View index {view_index} out of range for detections")
            return np.array(pil_image)
        frame_detections = detections[view_index]
    except (KeyError, IndexError, TypeError) as e:
        print(f"Warning: Invalid detection data format in {info_path}: {e}")
        return np.array(pil_image)
    
    # Calculate scaling factors for coordinate transformation
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Process each detection
    for detection in frame_detections:
        # Safely parse detection data
        try:
            class_id = int(detection[0])
            track_id = int(detection[1])
            x1, y1, x2, y2 = map(float, detection[2:6])
        except (ValueError, IndexError, TypeError):
            print(f"Warning: Invalid detection format: {detection}")
            continue
        
        # Only draw kart objects (class_id = 1)
        if class_id != 1:
            continue
        
        # Scale coordinates to current image size
        scaled_coords = (
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y)
        )
        x1_scaled, y1_scaled, x2_scaled, y2_scaled = scaled_coords
        
        # Filter out boxes that are too small
        box_width = x2_scaled - x1_scaled
        box_height = y2_scaled - y1_scaled
        if box_width < min_box_size or box_height < min_box_size:
            continue
        
        # Filter out boxes that are completely outside image bounds
        if (x2_scaled < 0 or x1_scaled > img_width or 
            y2_scaled < 0 or y1_scaled > img_height):
            continue
        
        # Determine color and draw bounding box
        color = (255, 0, 0) if track_id == 0 else COLORS.get(class_id, (255, 255, 255))
        draw.rectangle(
            [(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], 
            outline=color, 
            width=thickness
        )
    
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    try:
        with open(info_path) as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    try:
        detections = info["detections"][view_index]
    except IndexError:
        return []

    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    image_center_x = img_width / 2
    image_center_y = img_height / 2

    valid_karts = []
    closest_distance = float('inf')

    for detection in detections:
        object_class, track_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = map(int, detection[:6])

        if object_class != 1:
            continue

        scaled_x1 = bbox_x1 * scale_x
        scaled_y1 = bbox_y1 * scale_y
        scaled_x2 = bbox_x2 * scale_x
        scaled_y2 = bbox_y2 * scale_y

        box_width = scaled_x2 - scaled_x1
        box_height = scaled_y2 - scaled_y1

        if (box_width < min_box_size or box_height < min_box_size or
            scaled_x2 < 0 or scaled_x1 > img_width or
            scaled_y2 < 0 or scaled_y1 > img_height):
            continue

        center_x = (scaled_x1 + scaled_x2) / 2
        center_y = (scaled_y1 + scaled_y2) / 2
        center_point = (center_x, center_y)

        distance_from_center = (center_x - image_center_x)**2 + (center_y - image_center_y)**2

        instance_data = info.get("instances", info.get("karts", {}))
        if isinstance(instance_data, list):
            kart_name = instance_data[track_id] if track_id < len(instance_data) else f"kart_{track_id}"
        else:
            kart_name = instance_data.get(str(track_id), f"kart_{track_id}")

        kart_data = {
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": center_point
        }

        if distance_from_center < closest_distance:
            closest_distance = distance_from_center
            closest_kart = kart_data

        valid_karts.append(kart_data)

    if valid_karts and 'closest_kart' in locals():
        for kart in valid_karts:
            kart["is_center_kart"] = (kart["instance_id"] == closest_kart["instance_id"])

    return valid_karts


def extract_track_info(info_path: str) -> str:
    try:
        with open(info_path) as f:
            info = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return "Unknown Track"

    track_info = info.get("track", {})
    if isinstance(track_info, str):
        return track_info
    elif isinstance(track_info, dict):
        return track_info.get("name", "Unknown Track")
    else:
        return "Unknown Track"  


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    from collections import defaultdict

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next(kart for kart in karts if kart["is_center_kart"])
    track_name = extract_track_info(info_path)

    ego_center_x, ego_center_y = ego_kart["center"]
    other_karts = [kart for kart in karts if not kart["is_center_kart"]]

    position_distribution = defaultdict(int)
    qa_pairs = [
        {"question": "What kart is the ego car?", "answer": ego_kart["kart_name"]},
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts))},
        {"question": "What track is this?", "answer": track_name}
    ]

    for kart in other_karts:
        kart_center_x, kart_center_y = kart["center"]

        horizontal_relation = "left" if kart_center_x < ego_center_x else "right"
        vertical_relation = "front" if kart_center_y < ego_center_y else "back"

        position_distribution[horizontal_relation] += 1
        position_distribution[vertical_relation] += 1

        qa_pairs.extend([
            {
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": horizontal_relation
            },
            {
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": vertical_relation
            },
            {
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": f"{vertical_relation} and {horizontal_relation}"
            }
        ])

    qa_pairs.extend([
        {"question": "How many karts are to the left of the ego car?", "answer": str(position_distribution["left"])},
        {"question": "How many karts are to the right of the ego car?", "answer": str(position_distribution["right"])},
        {"question": "How many karts are in front of the ego car?", "answer": str(position_distribution["front"])},
        {"question": "How many karts are behind the ego car?", "answer": str(position_distribution["back"])}
    ])

    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    qa_pairs = generate_qa_pairs(info_file, view_index)

    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)



def generate_all(
    input_dir: str = "data/train",
    output_file: str | None = None,
    max_views: int = 10,
) -> int:
    input_dir = Path(input_dir)
    split_name = input_dir.name

    if output_file is None:
        output_file = input_dir / f"{split_name}_qa_pairs.json"
    else:
        output_file = Path(output_file)

    qa_dataset: List[Dict[str, str]] = []

    info_files = list(input_dir.glob("*_info.json"))
    for info_path in tqdm(info_files, desc=f"Building QA pairs for {split_name}"):
        stub = info_path.stem.replace("_info", "")
        for view in range(max_views):
            try:
                pairs = generate_qa_pairs(str(info_path), view)
            except Exception:
                continue

            if not pairs:
                continue

            image_stub = f"{split_name}/{stub}_{view:02d}_im.jpg"
            for qa in pairs:
                qa["image_file"] = image_stub
            qa_dataset.extend(pairs)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(qa_dataset, indent=2))

    print(f"\nSaved {len(qa_dataset):,} QA pairs → {output_file}")
    return len(qa_dataset)

def main():
    fire.Fire(
        {
            "check": check_qa_pairs,
            "generate_all": generate_all,   
        }
    )

if __name__ == "__main__":
    main()
