# @title Create data_split_game reference images (no augmentation)
# Change N_DATA_SPLIT_GAME_IMAGES in the control panel to choose how many files to create.

def export_data_split_game_images(samples, output_dir, n_images=30,
                                  n_same_person_images=2,
                                  image_size=IMG_SIZE_PRETRAINED,
                                  figsize_cm=7, dpi=300):
    output_dir = Path(output_dir)
    if output_dir.exists():
      output_dir.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert cm to inches for matplotlib.
    figsize_in = figsize_cm / 2.54

    # Keep reruns honest: delete previously generated images from this cell.
    for old_file in output_dir.glob("*.png"):
        old_file.unlink()

    if n_images is None:
        n_images = len(samples)

    n_images = int(n_images)
    n_same_person_images = int(n_same_person_images)

    if n_images <= 0:
        raise ValueError("N_DATA_SPLIT_GAME_IMAGES must be at least 1")

    if n_same_person_images < 1:
        raise ValueError("n_same_person_images must be at least 1")

    if n_same_person_images > n_images:
        raise ValueError("n_same_person_images cannot be larger than n_images")

    def _person_id(image_path):
        parts = Path(image_path).stem.split("_")
        return "_".join(parts[:-1]) if len(parts) > 1 else Path(image_path).stem

    def _safe_filename(text):
        # Keeps names readable while avoiding problematic filename characters.
        return "".join(
            char if char.isalnum() or char in ("_", "-") else "_"
            for char in str(text)
        )

    person_to_samples = {}
    for image_path, label in samples:
        person_to_samples.setdefault(_person_id(image_path), []).append((image_path, label))

    rng = np.random.default_rng(SEED)

    # Pick one person who has enough images to be repeated.
    repeat_candidates = [
        person_id
        for person_id, person_samples in person_to_samples.items()
        if len(person_samples) >= n_same_person_images
    ]

    if len(repeat_candidates) == 0:
        raise ValueError(
            f"No person has at least {n_same_person_images} images available."
        )

    repeat_candidates = np.array(repeat_candidates)
    repeat_counts = np.array(
        [len(person_to_samples[p]) for p in repeat_candidates],
        dtype=float
    )
    repeat_weights = repeat_counts / repeat_counts.sum()

    repeated_person = rng.choice(
        repeat_candidates,
        size=1,
        replace=False,
        p=repeat_weights
    )[0]

    selected_items = []

    # Select multiple different images from the repeated person.
    repeated_samples = person_to_samples[repeated_person]
    repeated_indices = rng.choice(
        len(repeated_samples),
        size=n_same_person_images,
        replace=False
    )

    for same_rank, sample_idx in enumerate(repeated_indices, start=1):
        image_path, label = repeated_samples[int(sample_idx)]
        selected_items.append({
            "person_id": repeated_person,
            "image_path": image_path,
            "label": label,
            "same_rank": same_rank,
            "is_repeated_person": True,
        })

    # Fill the remaining slots with one image per other person.
    remaining_slots = n_images - n_same_person_images

    other_people = [
        person_id
        for person_id in person_to_samples.keys()
        if person_id != repeated_person
    ]

    if remaining_slots > len(other_people):
        raise ValueError(
            f"Cannot create {n_images} images with only one repeated person and all "
            f"other people unique. Maximum possible is "
            f"{n_same_person_images + len(other_people)}."
        )

    if remaining_slots > 0:
        other_people = np.array(other_people)
        other_counts = np.array(
            [len(person_to_samples[p]) for p in other_people],
            dtype=float
        )
        other_weights = other_counts / other_counts.sum()

        chosen_other_people = rng.choice(
            other_people,
            size=remaining_slots,
            replace=False,
            p=other_weights
        )

        for person_id in chosen_other_people:
            person_samples = person_to_samples[person_id]
            sample_idx = int(rng.integers(len(person_samples)))
            image_path, label = person_samples[sample_idx]

            selected_items.append({
                "person_id": person_id,
                "image_path": image_path,
                "label": label,
                "same_rank": 1,
                "is_repeated_person": False,
            })

    # Shuffle so the repeated person's images are not necessarily next to each other.
    rng.shuffle(selected_items)

    saved_paths = []

    for rank, item in enumerate(selected_items, start=1):
        person_id = item["person_id"]
        image_path = item["image_path"]
        safe_person_id = _safe_filename(person_id)

        patch_id = f"{rank:03d}"
        same_rank = f"{item['same_rank']:02d}"

        # Load from the original file and apply only deterministic resize.
        patch = Image.open(image_path)
        patch = patch.resize((image_size, image_size), Image.BILINEAR)
        patch = np.array(patch)

        fig, ax = plt.subplots(figsize=(figsize_in, figsize_in), dpi=dpi)

        ax.imshow(patch, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

        ax.text(
            0.5, 0.03,
            "Image: LFW (Huang et al., 2007)",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=5,
            color="white",
            alpha=0.8,
            path_effects=[
                pe.withStroke(linewidth=1, foreground="black", alpha=0.3)
            ]
        )

        # Fill the whole 7x7 cm canvas without changing output size.
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if item["is_repeated_person"]:
            output_path = output_dir / (
                f"{safe_person_id}_REPEATED_{same_rank}_image_{patch_id}.png"
            )
        else:
            output_path = output_dir / (
                f"{safe_person_id}_image_{patch_id}.png"
            )

        fig.savefig(
            output_path,
            dpi=dpi,
            pad_inches=0
        )
        plt.close(fig)

        saved_paths.append(output_path)

    return saved_paths  

DATA_SPLIT_GAME_DIR = "/content/drive/MyDrive/cvlab_workshop/data_split_game"
N_DATA_SPLIT_GAME_IMAGES = 200

# Force exactly this many images from the same person.
N_SAME_PERSON_IMAGES = 5

data_split_game_paths = export_data_split_game_images(
    full_train_display.samples,
    DATA_SPLIT_GAME_DIR,
    n_images=N_DATA_SPLIT_GAME_IMAGES,
    n_same_person_images=N_SAME_PERSON_IMAGES,
)

print(f"Saved {len(data_split_game_paths)} no-augmentation input patches to {DATA_SPLIT_GAME_DIR}")