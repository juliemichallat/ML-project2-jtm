def get_hf_by_filename(dataframe, target_frame_idx):
    # On cherche la ligne où la colonne 'frame_idx' correspond exactement
    idx_numerique = np.int64(target_frame_idx)
    result = dataframe.loc[dataframe['frame_idx'] == idx_numerique, 'HF']

    # Si on trouve un résultat (la liste n'est pas vide)
    if not result.empty:
        return result.values[0] # On retourne la première valeur trouvée
    else:
        return None # Aucun fichier trouvé

def get_subject_dict(dataframe, subject_id):
    # On filtre la ligne où la colonne 'Subject' correspond à l'ID (ex: E14)
    row = dataframe[dataframe['Subject'] == subject_id]

    if not row.empty:
        # .iloc[0] prend la première ligne trouvée
        # .to_dict() convertit cette ligne en dictionnaire
        return row.iloc[0].to_dict()
    else:
        return None
def get_subject_protocol_img_from_folder(file):

  stem = Path(file).stem
  parts = stem.split("-")

  patient_id = parts[0]
  protocol   = parts[1]

  return patient_id, protocol

def get_frame_idx_img_from_filename(file):

  stem = Path(file).stem
  parts = stem.split("_")

  frame_idx  = parts[1]

  return frame_idx

def process_thermal_background(image_path, seg_model_instance, conf_threshold=0.5):
    # 1. Chargement (Gère le float32)
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None: return None, None

    # 2. Normalisation robuste (Float32 -> Uint8 pour YOLO)
    # On trouve le min et max de l'image pour étaler le contraste
    img_norm = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX)

    # Conversion explicite en entiers 8-bits
    img_8bit = img_norm.astype(np.uint8)

    # Passage en 3 couches (Couleur) pour YOLO
    if len(img_8bit.shape) == 2:
        img_8bit_color = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        img_8bit_color = img_8bit

    # 3. Inférence YOLO
    results = seg_model_instance.predict(source=img_8bit_color, conf=conf_threshold, classes=[0], retina_masks=True, verbose=False)
    result = results[0]

    # 4. Création du Masque
    h, w = img_raw.shape[:2]
    person_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is not None:
        for mask_contour in result.masks.xy:
            if len(mask_contour) > 0:
                cv2.fillPoly(person_mask, [mask_contour.astype(np.int32)], 1)

        # --- ATTENTION ICI ---
        # Pour float32, on met 0.0 (float) et non 0 (int), même si Python gère souvent les deux
        # On copie l'image pour ne pas modifier l'original si besoin
        img_processed = img_raw.copy()
        img_processed[person_mask == 1] = 0.0
    else:
        img_processed = img_raw

    # 5. Calcul des stats (sur les valeurs float32 d'origine !)
    background_pixels = img_processed[img_processed > 0] # On ignore les 0.0 qu'on vient de mettre

    if len(background_pixels) == 0: return img_processed, None

    stats = {
        "mean_temp_background": np.mean(background_pixels),
        "min_temp_background": np.min(background_pixels),
        "max_temp_background": np.max(background_pixels),
        "std_dev_temp_background": np.std(background_pixels),
        "pixel_count_background": len(background_pixels)
    }

    return img_processed, stats


def extract_chest_geometric(image_path, seg_model_instance, pose_model_instance, chest_ratio=0.7):
    # 1. Chargement & Normalisation
    img_raw = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_raw is None: return None, None

    # Normalisation Float32 -> Uint8
    img_norm = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX)
    img_8bit = img_norm.astype(np.uint8)

    if len(img_8bit.shape) == 2:
        img_ai = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2BGR)
    else:
        img_ai = img_8bit

    h, w = img_ai.shape[:2]

    # 2. Segmentation
    results_seg = seg_model_instance.predict(img_ai, classes=[0], retina_masks=True, verbose=False, conf=0.5)
    if results_seg[0].masks is None: return None, None

    person_mask = np.zeros((h, w), dtype=np.uint8)
    for mask_contour in results_seg[0].masks.xy:
        if len(mask_contour) > 0:
            cv2.fillPoly(person_mask, [mask_contour.astype(np.int32)], 255)

    # 3. Pose & Géométrie
    results_pose = pose_model_instance.predict(img_ai, verbose=False)
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    if results_pose[0].keypoints is not None:
        kpts = results_pose[0].keypoints.xy.cpu().numpy()[0]
        ls, rs = kpts[5], kpts[6]

        if not (np.any(ls == 0) or np.any(rs == 0)):
            shoulder_width = np.linalg.norm(ls - rs)
            y_top = int(min(ls[1], rs[1]))
            box_height = int(shoulder_width * chest_ratio)
            y_bottom = y_top + box_height
            padding = int(shoulder_width * 0.15)
            x_left = int(min(ls[0], rs[0])) + padding
            x_right = int(max(ls[0], rs[0])) - padding

            cv2.rectangle(roi_mask, (x_left, y_top), (x_right, y_bottom), 255, -1)
        else:
            return None, None
    else:
        return None, None

    # 4. Intersection et Extraction
    final_mask = cv2.bitwise_and(person_mask, roi_mask)

    # On extrait les pixels du RAW (float32)
    chest_pixels = img_raw[final_mask == 255]

    # Visuel (Optionnel, juste pour voir le résultat en 8-bits)
    visual_result = cv2.bitwise_and(img_ai, img_ai, mask=final_mask)

    stats_chest = None
    if len(chest_pixels) > 0:
        stats_chest = {
            "mean_temp_chest": np.mean(chest_pixels),
            "stdt_temp_chest": np.std(chest_pixels),
            "min_temp_chest": np.min(chest_pixels),
            "max_temp_chest": np.max(chest_pixels),
            "pixel_count_chest": len(chest_pixels)
        }

    return visual_result, stats_chest

def create_dataset_from_folder(folder_path, physio_path, subject_path, output_name):

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Accélération activée sur : {device.upper()}")

    # --- A. Load Reference DataFrames ---
    print("Loading reference CSVs...")
    try:
        # Assuming physio CSV has columns 'frame_idx' and 'HF'
        df_physio = pd.read_csv(physio_path)
        # Assuming subject CSV has column 'Subject'
        df_subjects = pd.read_csv(subject_path)
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        # We create dummy DFs just so the script doesn't crash if files are missing
        df_physio = pd.DataFrame(columns=['frame_idx', 'HF'])
        df_subjects = pd.DataFrame(columns=['Subject'])

    # --- B. Get Folder Details ---
    # We use the folder name "E14-P2" to get the Subject and Protocol
    folder_name = Path(folder_path).name
    try:
        # Using your provided function to parse the FOLDER name
        patient_id, protocol = get_subject_protocol_img_from_folder(folder_name)
    except Exception as e:
        print(f"Error parsing folder name '{folder_name}': {e}")
        return

    # --- C. List Images ---
    # Looking for image files inside the folder
    valid_exts = ('.tif', '.tiff', '.png', '.jpg')
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])

    print("Chargement des modèles IA en mémoire (GPU)...")
    # Utiliser .to('cuda') force l'utilisation du GPU si dispo
    seg_model_loaded = YOLO('yolov8l-seg.pt')
    pose_model_loaded = YOLO('yolo11l-pose.pt') # Ou yolov8l-pose.pt
    print("Modèles chargés !")

    print(f"Processing {len(image_files)} images for Subject: {patient_id}, Protocol: {protocol}")

    data_rows = []

    # --- D. Loop Through Images ---
    for filename in tqdm(image_files):
        full_image_path = os.path.join(folder_path, filename)

        # 1. Get Frame Index
        try:
            # Using your provided function
            frame_idx = get_frame_idx_img_from_filename(filename)
        except:
            print(f"Skipping {filename}: name format invalid")
            continue

        # 2. Initialize Row Data
        row = {
            "filename": filename,
            "patient_id": patient_id,
            "protocol": protocol,
            "frame_idx": frame_idx
        }

        # 3. Get Physiological Data (HF)
        hf_val = get_hf_by_filename(df_physio, frame_idx)
        row["HF"] = hf_val

        # 4. Get Subject Metadata (Age, Gender, etc)
        subj_meta = get_subject_dict(df_subjects, patient_id)
        if subj_meta:
            # --- THE FIX IS HERE ---
            # Remove 'Subject' from the dictionary if it exists
            # because we already have 'patient_id' in the row
            if 'Subject' in subj_meta:
                del subj_meta['Subject']

            row.update(subj_meta)

        # 5. Extract Background Stats
        # Using your provided function
        background_pixels, bg_stats = process_thermal_background(full_image_path,seg_model_instance=seg_model_loaded)

        if bg_stats:
            row.update(bg_stats)
        else:
            # Fill with None if detection failed
            row.update({
                "mean_temp_background": None, "min_temp_background": None,
                "max_temp_background": None, "std_dev_temp_background": None,
                "pixel_count_background": None
            })

        # 6. Extract Chest Stats
        # Using your provided function
        chest_pixels, chest_stats = extract_chest_geometric(full_image_path, seg_model_instance=seg_model_loaded, pose_model_instance=pose_model_loaded)

        # Note: This will likely be None due to the return bug mentioned above
        if chest_stats:
            row.update(chest_stats)
        else:
            row.update({
                "mean_temp_chest": None, "stdt_temp_chest": None,
                "min_temp_chest": None, "max_temp_chest": None,
                "pixel_count_chest": None
            })

        data_rows.append(row)

    # --- E. Save to CSV ---
    if len(data_rows) > 0:
        final_df = pd.DataFrame(data_rows)

        # Optional: Reorder columns for readability
        cols = ['filename','patient_id', 'protocol', 'frame_idx', 'HF']
        remaining = [c for c in final_df.columns if c not in cols]
        final_df = final_df[cols + remaining]

        final_df.to_csv(output_name, index=False, sep = ";")
        print(f"\nDataset saved successfully to: {output_name}")
        print(f"Total Rows: {len(final_df)}")
        print(final_df.head())
    else:
        print("No data processed.")

def create_dataset(root_data_folder, aligned_root_folder, subject_csv_path, final_output_path):
    """
    Iterates through folders, extracts subfolder P#, and matches CSVs
    with specific casing (e.g., U13_p4_aligned.csv).
    """
    all_dfs = []

    # Filter out hidden folders like .ipynb_checkpoints
    data_subfolders = sorted([
        f for f in os.listdir(root_data_folder)
        if os.path.isdir(os.path.join(root_data_folder, f)) and not f.startswith('.')
    ])

    for folder_name in data_subfolders:
        # Split 'U13-P4' into ['U13', 'P4']
        parts = folder_name.split('-')
        if len(parts) < 2:
            print(f"Skipping {folder_name}: Doesn't match expected format.")
            continue

        subject_id = parts[0]  # e.g., 'U13'
        p_suffix = parts[1]    # e.g., 'P4'

        # Construct filename: First part capital, second part lowercase
        # Result: U13_p4_aligned.csv
        csv_filename = f"{subject_id}_{p_suffix.lower()}_aligned.csv"

        # Path: .../aligned/P4/U13_p4_aligned.csv
        current_physio_path = os.path.join(aligned_root_folder, p_suffix, csv_filename)
        current_folder_path = os.path.join(root_data_folder, folder_name)

        print(f"Looking for: {csv_filename} in subfolder {p_suffix}...")

        if not os.path.exists(current_physio_path):
            print(f"Warning: File NOT found: {current_physio_path}")
            continue

        temp_csv = f"temp_{folder_name}.csv"

        try:
            # Call your existing function
            create_dataset_from_folder(current_folder_path, current_physio_path, subject_csv_path, temp_csv)

            if os.path.exists(temp_csv):
                df = pd.read_csv(temp_csv)
                all_dfs.append(df)
                os.remove(temp_csv)
                print(f"Successfully processed {folder_name} ({len(df)} rows)")

        except Exception as e:
            print(f"Error processing {folder_name}: {e}")

    # Final Concatenation
    if all_dfs:
        final_dataset = pd.concat(all_dfs, ignore_index=True)
        final_dataset.to_csv(final_output_path, index=False)
        print(f"\nALL DONE! Master dataset saved to: {final_output_path}")
        print(f"Total combined rows: {len(final_dataset)}")
    else:
        print("\nNo dataframes were created. Check the folder names and CSV casing.")
