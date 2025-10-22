from pathlib import Path

import fire
from matplotlib import pyplot as plt

from .generate_qa import draw_detections, extract_frame_info, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    try:
        with open(info_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    karts = extract_kart_objects(info_path, view_index, img_width, img_height)
    if not karts:
        return []

    ego_kart = next(kart for kart in karts if kart["is_center_kart"])
    track_name = extract_track_info(info_path)

    ego_name = ego_kart["kart_name"]
    ego_center_x, ego_center_y = ego_kart["center"]
    other_karts = [kart for kart in karts if not kart["is_center_kart"]]

    captions = []
    other_count = len(other_karts)

    # Diverse ego-centric descriptions
    ego_descriptions = [
        f"From {ego_name}'s perspective, racing on the {track_name} track.",
        f"{ego_name} leads the race on the {track_name} circuit.",
        f"The race view from {ego_name}'s kart on the {track_name} track.",
        f"{ego_name} navigates the {track_name} track in this racing scene.",
    ]
    captions.extend(ego_descriptions[:2])  # Use first 2 for variety

    # Rich positional descriptions with racing context
    position_templates = [
        "{other_name} is racing {vertical_pos} and to the {horizontal_pos} of {ego_name}",
        "{other_name} appears {vertical_pos} and {horizontal_pos} from {ego_name}'s viewpoint",
        "In the {horizontal_pos}, {other_name} is positioned {vertical_pos} of {ego_name}",
        "{ego_name} can see {other_name} {vertical_pos} and to the {horizontal_pos}",
    ]

    for i, other_kart in enumerate(other_karts):
        other_name = other_kart["kart_name"]
        other_x, other_y = other_kart["center"]
        
        horizontal_pos = "left" if other_x < ego_center_x else "right"
        vertical_pos = "ahead" if other_y < ego_center_y else "behind"
        
        # Use different template for variety
        template = position_templates[i % len(position_templates)]
        captions.append(template.format(
            other_name=other_name,
            ego_name=ego_name,
            horizontal_pos=horizontal_pos,
            vertical_pos=vertical_pos
        ))

    # Enhanced counting descriptions
    if other_count == 0:
        captions.extend([
            f"{ego_name} races alone on the {track_name} track.",
            f"The {track_name} track is clear ahead for {ego_name}.",
        ])
    elif other_count == 1:
        other_name = other_karts[0]["kart_name"]
        other_x, other_y = other_karts[0]["center"]
        horizontal_pos = "left" if other_x < ego_center_x else "right"
        vertical_pos = "ahead" if other_y < ego_center_y else "behind"
        
        captions.extend([
            f"{ego_name} races against one opponent on the {track_name} track.",
            f"Only {other_name} is visible {vertical_pos} and to the {horizontal_pos}.",
            f"The competition includes {ego_name} and {other_name} on the {track_name} circuit.",
        ])
    else:
        # Multiple competitors - more strategic descriptions
        ahead_count = sum(1 for k in other_karts if k["center"][1] < ego_center_y)
        behind_count = other_count - ahead_count
        
        captions.extend([
            f"{ego_name} is racing with {other_count} competitors on the {track_name} track.",
            f"{ahead_count} karts are ahead while {behind_count} trail behind {ego_name}.",
            f"The {track_name} track is busy with {ego_name} and {other_count} other racers.",
        ])

    # Add track-specific context when available
    if track_name != "Unknown Track":
        captions.extend([
            f"{ego_name} competes on the challenging {track_name} circuit.",
            f"The {track_name} track conditions favor {ego_name}'s racing line.",
        ])

    # Summary position overview (enhanced)
    if other_count > 1:
        ahead_karts = [k for k in other_karts if k["center"][1] < ego_center_y]
        behind_karts = [k for k in other_karts if k["center"][1] >= ego_center_y]
        left_karts = [k for k in other_karts if k["center"][0] < ego_center_x]
        right_karts = [k for k in other_karts if k["center"][0] >= ego_center_x]
        
        position_summary = []
        if ahead_karts:
            ahead_names = [k["kart_name"] for k in ahead_karts]
            position_summary.append(f"ahead: {', '.join(ahead_names)}")
        if behind_karts:
            behind_names = [k["kart_name"] for k in behind_karts]
            position_summary.append(f"behind: {', '.join(behind_names)}")
        if left_karts:
            left_names = [k["kart_name"] for k in left_karts]
            position_summary.append(f"left: {', '.join(left_names)}")
        if right_karts:
            right_names = [k["kart_name"] for k in right_karts]
            position_summary.append(f"right: {', '.join(right_names)}")
            
        if position_summary:
            summary_text = "; ".join(position_summary)
            captions.append(f"Racing situation overview - {summary_text}")

    return captions


def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""

import json
from pathlib import Path
from tqdm import tqdm
import fire

def generate_all(
    input_dir: str = "data/train",
    output_file: str = "data/train/train_captions.json",
    max_views: int = 10,
):
    root = Path(input_dir).resolve()
    records = []

    for info_path in tqdm(root.rglob("*_info.json"), desc="Processing JSON files"):
        stem = info_path.stem.removesuffix("_info")

        for view in range(max_views):
            captions = generate_caption(str(info_path), view)
            if not captions:
                continue

            jpg_path = info_path.parent / f"{stem}_{view:02d}_im.jpg"
            rel_path = Path("train") / jpg_path.relative_to(root)

            records.extend(
                {"image_file": str(rel_path), "caption": caption}
                for caption in captions
            )

    Path(output_file).write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records):,} captions → {output_file}")


def main():
    fire.Fire(
        {
            "check": check_caption,
            "generate_all": generate_all,   
        }
    )


if __name__ == "__main__":
    main()
