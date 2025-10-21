import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

# Define object type mapping
OBJECT_TYPES = {
    1: "Kart",
    2: "Track Boundary",
    3: "Track Element",
    4: "Special Element 1",
    5: "Special Element 2",
    6: "Special Element 3",
}

# Define colors for different object types (RGB format)
COLORS = {
    1: (0, 255, 0),  # Green for karts
    2: (255, 0, 0),  # Blue for track boundaries
    3: (0, 0, 255),  # Red for track elements
    4: (255, 255, 0),  # Cyan for special elements
    5: (255, 0, 255),  # Magenta for special elements
    6: (0, 255, 255),  # Yellow for special elements
}

# Original image dimensions for the bounding box coordinates
ORIGINAL_WIDTH = 600
ORIGINAL_HEIGHT = 400


def extract_frame_info(image_path: str) -> tuple[int, int]:
    """
    Extract frame ID and view index from image filename.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (frame_id, view_index)
    """
    filename = Path(image_path).name
    # Format is typically: XXXXX_YY_im.png where XXXXX is frame_id and YY is view_index
    parts = filename.split("_")
    if len(parts) >= 2:
        frame_id = int(parts[0], 16)  # Convert hex to decimal
        view_index = int(parts[1])
        return frame_id, view_index
    return 0, 0  # Default values if parsing fails


def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
) -> np.ndarray:
    """
    Draw detection bounding boxes and labels on the image.

    Args:
        image_path: Path to the image file
        info_path: Path to the corresponding info.json file
        font_scale: Scale of the font for labels
        thickness: Thickness of the bounding box lines
        min_box_size: Minimum size for bounding boxes to be drawn

    Returns:
        The annotated image as a numpy array
    """
    # Read the image using PIL
    pil_image = Image.open(image_path)
    if pil_image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Get image dimensions
    img_width, img_height = pil_image.size

    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)

    # Extract frame ID and view index from image filename
    _, view_index = extract_frame_info(image_path)

    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)

    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT

    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)

        if class_id != 1:
            continue

        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue

        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue

        # Get color for this object type
        if track_id == 0:
            color = (255, 0, 0)
        else:
            color = COLORS.get(class_id, (255, 255, 255))

        # Draw bounding box using PIL
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)

    # Convert PIL image to numpy array for matplotlib
    return np.array(pil_image)


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    # TODO 

    with open(info_path) as f:
        info = json.load(f)
    
    try:
        detections = info["detections"][view_index]
    except IndexError:
        return []

    # calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    img_center = (img_width / 2, img_height / 2)
    karts = []

    for detection in detections:
        class_id, track_id, x1, y1, x2, y2 = map(int, detection[:6])
        
        # filter non-karts
        if class_id != 1:
            continue
            
        # coordinates and dimensions
        x1s = x1 * scale_x
        y1s = y1 * scale_y
        x2s = x2 * scale_x
        y2s = y2 * scale_y
        width = x2s - x1s
        height = y2s - y1s
        
        # validate bounding box
        if (width < min_box_size or height < min_box_size or
            x2s < 0 or x1s > img_width or
            y2s < 0 or y1s > img_height):
            continue
        
        # center point
        center = ((x1s + x2s) / 2, (y1s + y2s) / 2)
        
        # kart name from metadata
        instance_data = info.get("instances", info.get("karts", {}))
        if isinstance(instance_data, list):
            kart_name = instance_data[track_id] if track_id < len(instance_data) else f"kart_{track_id}"
        else:
            kart_name = instance_data.get(str(track_id), f"kart_{track_id}")
        
        karts.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": center
        })

    # find ego kart that is closest to image center
    if karts:
        ego_kart = min(karts, key=lambda k: 
            (k["center"][0] - img_center[0])**2 + 
            (k["center"][1] - img_center[1])**2
        )
        for kart in karts:
            kart["is_center_kart"] = (kart["instance_id"] == ego_kart["instance_id"])
    
    return karts


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    # TODO

    with open(info_path) as f:
        info = json.load(f)

    track_info = info.get("track", {})
    if isinstance(track_info, str):
        return track_info  
    elif isinstance(track_info, dict):
        return track_info.get("name", "Unknown Track")  
    else:
        return "Unknown Track"  


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # 1. Ego car question
    # What kart is the ego car?

    # 2. Total karts question
    # How many karts are there in the scenario?

    # 3. Track information questions
    # What track is this?

    # 4. Relative position questions for each kart
    # Is {kart_name} to the left or right of the ego car?
    # Is {kart_name} in front of or behind the ego car?
    # Where is {kart_name} relative to the ego car?

    # 5. Counting questions
    # How many karts are to the left of the ego car?
    # How many karts are to the right of the ego car?
    # How many karts are in front of the ego car?
    # How many karts are behind the ego car?

    # TODO 

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    # ego kart (center kart) and track name
    ego = next(k for k in karts if k["is_center_kart"])
    track_name = extract_track_info(info_path)
    ego_x, ego_y = ego["center"]
    
    # init counters and QA list
    position_counts = {"left": 0, "right": 0, "front": 0, "back": 0}
    qa_pairs = [
        {"question": "What kart is the ego car?", "answer": ego["kart_name"]},
        {"question": "How many karts are there in the scenario?", "answer": str(len(karts))},
        {"question": "What track is this?", "answer": track_name}
    ]
    
    # process each non-ego kart
    for kart in karts:
        if kart["instance_id"] == ego["instance_id"]:
            continue
            
        x, y = kart["center"]
        relative_position = {
            "horizontal": "left" if x < ego_x else "right",
            "vertical": "front" if y < ego_y else "back"
        }
        
        # update counters
        position_counts[relative_position["horizontal"]] += 1
        position_counts[relative_position["vertical"]] += 1
        
        # kart-specific questions
        qa_pairs.extend([
            {
                "question": f"Is {kart['kart_name']} to the left or right of the ego car?",
                "answer": relative_position["horizontal"]
            },
            {
                "question": f"Is {kart['kart_name']} in front of or behind the ego car?",
                "answer": relative_position["vertical"]
            },
            {
                "question": f"Where is {kart['kart_name']} relative to the ego car?",
                "answer": f"{relative_position['vertical']} and {relative_position['horizontal']}"
            }
        ])
    
    qa_pairs.extend([
        {"question": "How many karts are to the left of the ego car?", "answer": str(position_counts["left"])},
        {"question": "How many karts are to the right of the ego car?", "answer": str(position_counts["right"])},
        {"question": "How many karts are in front of the ego car?", "answer": str(position_counts["front"])},
        {"question": "How many karts are behind the ego car?", "answer": str(position_counts["back"])}
    ])
    
    return qa_pairs


def check_qa_pairs(info_file: str, view_index: int):
    """
    Check QA pairs for a specific info file and view index.

    Args:
        info_file: Path to the info.json file
        view_index: Index of the view to analyze
    """
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)

    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)

    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_qa.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

from pathlib import Path
import json
from typing import List, Dict
from tqdm import tqdm 

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
