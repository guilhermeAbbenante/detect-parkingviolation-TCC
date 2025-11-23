import cv2
import numpy as np
from ultralytics import YOLO
from math import sqrt
import os
from typing import List, Dict, Tuple, Set, Any

# ==============================================================================
# CONFIGURAÇÃO
# ==============================================================================
MODEL_PATH = 'model/best.pt'
SOURCE_FOLDER = 'imagens'
OUTPUT_FOLDER = 'resultados'

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

CLASS_NAMES = {
    0: 'calcada', 1: 'carro', 2: 'faixa_pedestre', 3: 'guia_amarela',
    4: 'guia_normal', 5: 'guia_rebaixada', 6: 'placa_proibido', 7: 'rampa', 8: 'rua'
}

VEHICLE_CLASS = 'carro'
RELATIONAL_CLASS = 'placa_proibido'

PROHIBITED_ZONES = [
    'calcada', 'faixa_pedestre', 'guia_amarela', 'guia_rebaixada', 'rampa'
]

THRESHOLDS = {
    'calcada': 50,
    'faixa_pedestre': 15,
    'guia_amarela': 50,
    'guia_rebaixada': 50,
    'rampa': 50
}

#Parametros
DYNAMIC_DISTANCE_FACTOR = 2.0
MIN_RATIO_CAR_PLACA = 5.0
MAX_RATIO_CAR_PLACA = 100.0
CONTACT_MASK_PERCENTAGE = 0.60 

# ==============================================================================
# FUNÇÕES UTILITÁRIAS (GEOMETRIA E MASCARAS)
# ==============================================================================

def get_bottom_mask(full_mask: np.ndarray, percentage: float) -> np.ndarray:
    """Retorna apenas a porcentagem inferior da máscara. Usado para verificar infrações de faixa de pedestre."""
    rows = np.any(full_mask, axis=1)
    if not np.any(rows):
        return full_mask
    
    y_indices = np.where(rows)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    height = y_max - y_min
    
    cut_y = int(y_max - (height * percentage))
    
    bottom_mask = np.zeros_like(full_mask)
    bottom_mask[cut_y:y_max+1, :] = full_mask[cut_y:y_max+1, :]
    return bottom_mask

def get_center(bbox: List[int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_box_area(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def get_width(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return x2 - x1

def is_near(box1: List[int], box2: List[int], max_dist: float) -> Tuple[bool, float]:
    c1 = get_center(box1)
    c2 = get_center(box2)
    dist = sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    return dist < max_dist, dist

def get_violation_text(class_name: str) -> Tuple[str, str]:
    data = {
        'calcada': ("Estacionado na Calcada", "Grave"),
        'faixa_pedestre': ("Estacionado na Faixa", "Grave"),
        'guia_rebaixada': ("Obstruindo Garagem", "Média"),
        'guia_amarela': ("Local Proibido (Guia Amarela)", "Média"),
        'rampa': ("Estacionado na Rampa", "Grave"),
        'placa_proibido': ("Estacionado sob Placa Proibida", "Grave")
    }
    return data.get(class_name, (f"Estacionado em {class_name}", "Grave"))

def get_unique_filename(folder: str, base_name: str, ext: str) -> str:
    base_path = os.path.join(folder, f"{base_name}{ext}")
    if not os.path.exists(base_path): return base_path
    counter = 1
    while True:
        path = os.path.join(folder, f"{base_name}{counter}{ext}")
        if not os.path.exists(path): return path
        counter += 1

# ==============================================================================
# ANALISE DE INFRAÇÕES
# ==============================================================================

def parse_detections(results, w: int, h: int) -> Dict:
    """
    Extrai as informações brutas do YOLO e organiza em um dicionário.
    Separa carros, zonas proibidas e placas.
    """
    data = {
        'cars': [],
        'zones': {}, 
        'plates': [],
        'detected_classes': set()
    }

    if results.masks is None:
        return data

    for i, mask_tensor in enumerate(results.masks.data):
        class_id = int(results.boxes[i].cls[0])
        class_name = CLASS_NAMES.get(class_id, 'desconhecido')
        data['detected_classes'].add(class_name)
        
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        bbox = results.boxes[i].xyxy[0].cpu().numpy().astype(int)

        obj_data = {'bbox': bbox, 'mask': mask_resized}

        if class_name == VEHICLE_CLASS:
            data['cars'].append(obj_data)
        elif class_name in PROHIBITED_ZONES:
            if class_name not in data['zones']: data['zones'][class_name] = []
            data['zones'][class_name].append(mask_resized)
        elif class_name == RELATIONAL_CLASS:
            data['plates'].append(obj_data)
    
    return data

def check_plate_violations(car: Dict, plates: List[Dict]) -> bool:
    if not plates: return False
    
    car_w = get_width(car['bbox'])
    
    for plate in plates:
        plate_area = get_box_area(plate['bbox'])
        if plate_area < 1: continue
        
        ratio = car['area'] / plate_area
        if not (MIN_RATIO_CAR_PLACA <= ratio <= MAX_RATIO_CAR_PLACA):
            continue
            
        is_close, dist = is_near(car['bbox'], plate['bbox'], car_w * DYNAMIC_DISTANCE_FACTOR)
        
        if is_close:
            tipo, gravidade = get_violation_text(RELATIONAL_CLASS)
            car['infractions'].append({
                'class_name': RELATIONAL_CLASS,
                'tipo': tipo,
                'intensidade': gravidade,
                'detalhe': f"Prox. Placa (Dist: {dist:.0f}px)"
            })
            return True
    return False

def check_ground_violations(car: Dict, zones: Dict):
    is_faixa_infr = False
    
    #Usa a mascara de contato (faixa de pedestre)
    if 'faixa_pedestre' in zones:
        combined_zone = np.zeros_like(car['mask'], dtype=bool)
        for m in zones['faixa_pedestre']:
            combined_zone = np.logical_or(combined_zone, m)
        
        overlap = np.sum(np.logical_and(car['contact_mask'], combined_zone))
        
        if overlap > THRESHOLDS['faixa_pedestre']:
            is_faixa_infr = True
            tipo, grav = get_violation_text('faixa_pedestre')
            car['infractions'].append({
                'class_name': 'faixa_pedestre',
                'tipo': tipo,
                'intensidade': grav,
                'detalhe': f"Sobreposicao: {overlap} pixels em 'faixa_pedestre'"
            })

    #Usa a mascara completa (calçada)
    if not is_faixa_infr:
        max_overlap = 0
        worst_class = None
        
        for cls_name, masks in zones.items():
            if cls_name == 'faixa_pedestre': continue 
            
            threshold = THRESHOLDS.get(cls_name, 50)
            combined_zone = np.zeros_like(car['mask'], dtype=bool)
            for m in masks:
                combined_zone = np.logical_or(combined_zone, m)
            
            overlap = np.sum(np.logical_and(car['mask'], combined_zone))
            
            if overlap > threshold:
                if overlap > max_overlap:
                    max_overlap = overlap
                    worst_class = cls_name
        
        if worst_class:
            tipo, grav = get_violation_text(worst_class)
            car['infractions'].append({
                'class_name': worst_class,
                'tipo': tipo,
                'intensidade': grav,
                'detalhe': f"Sobreposicao: {max_overlap} pixels em '{worst_class}'"
            })

def analyze_infractions(data: Dict) -> List[Dict]:
    processed_cars = []
    
    for i, car in enumerate(data['cars']):
        car_info = {
            'id': i + 1,
            'bbox': car['bbox'],
            'mask': car['mask'],
            'contact_mask': get_bottom_mask(car['mask'], CONTACT_MASK_PERCENTAGE),
            'area': get_box_area(car['bbox']),
            'infractions': []
        }
        
        is_plate_infraction = check_plate_violations(car_info, data['plates'])
        
        if not is_plate_infraction:
            check_ground_violations(car_info, data['zones'])
            
        processed_cars.append(car_info)
        
    return processed_cars

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

def draw_visuals(frame: np.ndarray, cars: List[Dict], zones: Dict, plates: List[Dict]) -> Tuple[np.ndarray, str]:
    violators = [c for c in cars if c['infractions']]
    primary_id = None
    status_key = "OK"
    
    if violators:
        primary = max(violators, key=lambda c: c['area'])
        primary_id = primary['id']
        status_key = primary['infractions'][0]['class_name']

    for car in cars:
        is_primary = (car['id'] == primary_id)
        is_violator = bool(car['infractions'])
        
        if is_primary: color = (0, 0, 255)
        elif is_violator: color = (0, 255, 255)
        else: color = (0, 255, 0)

        line2 = "OK"
        line1 = ""
        if is_violator:
            infr = car['infractions'][0]
            line2 = f"INFRA: {infr['tipo']} ({infr['intensidade']})"
            line1 = infr['detalhe']

        if is_violator:
            overlay = frame.copy()
            overlay[car['mask']] = overlay[car['mask']] * 0.5 + np.array(color) * 0.5
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        x1, y1, x2, y2 = car['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, line2, (x1, y1 - 10), font, 0.6, color, 2)
        if line1:
            cv2.putText(frame, line1, (x1, y1 - 35), font, 0.5, color, 1)

    zone_colors = {
        'calcada': (0, 255, 255), 'faixa_pedestre': (255, 0, 255),
        'guia_amarela': (0, 165, 255), 'guia_rebaixada': (255, 255, 0),
        'rampa': (128, 0, 128), 'default': (255, 0, 0)
    }
    
    for cls_name, masks in zones.items():
        color = zone_colors.get(cls_name, zone_colors['default'])
        for m in masks:
            overlay = frame.copy()
            overlay[m] = overlay[m] * 0.5 + np.array(color) * 0.5
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    for p in plates:
        overlay = frame.copy()
        overlay[p['mask']] = overlay[p['mask']] * 0.5 + np.array((255, 0, 255)) * 0.5
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

    return frame, status_key

# ==============================================================================
# MAIN
# ==============================================================================

def process_image(frame: np.ndarray, model: YOLO) -> Tuple[np.ndarray, str]:
    h, w = frame.shape[:2]
    
    results = model(frame, verbose=False)[0]
    parsed_data = parse_detections(results, w, h)
    
    if not parsed_data['cars']:
        return frame, "NONECAR"

    processed_cars = analyze_infractions(parsed_data)
    final_frame, status_key = draw_visuals(frame, processed_cars, parsed_data['zones'], parsed_data['plates'])

    return final_frame, status_key

def main():
    print("Starting Batch Processing...")
    
    try:
        model = YOLO(MODEL_PATH)
        print(f"Model loaded: {MODEL_PATH}")
    except Exception as e:
        print(f"Critical Error: {e}")
        return

    if not os.path.exists(SOURCE_FOLDER):
        print(f"Error: Source folder '{SOURCE_FOLDER}' not found.")
        return

    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    except Exception as e:
        print(f"Error creating output folder: {e}")
        return

    all_files = os.listdir(SOURCE_FOLDER)
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS]

    if not image_files:
        print(f"No valid images found in '{SOURCE_FOLDER}'.")
        return

    print(f"Found {len(image_files)} images.\n")
    count_success = 0
    
    for i, filename in enumerate(image_files):
        source_path = os.path.join(SOURCE_FOLDER, filename)
        
        try:
            print(f"[{i+1}/{len(image_files)}] Processing: {filename}...")
            
            frame = cv2.imread(source_path)
            if frame is None:
                print(f"  -> Error: Failed to read image.")
                continue

            result_frame, status_key = process_image(frame, model)
            
            file_ext = os.path.splitext(filename)[1].lower()
            base_name = f"detec_{status_key}"
            output_path = get_unique_filename(OUTPUT_FOLDER, base_name, file_ext)
            
            cv2.imwrite(output_path, result_frame)
            print(f"  -> Saved: {os.path.basename(output_path)}")
            count_success += 1
            
        except Exception as e:
            print(f"  -> Unexpected error in {filename}: {e}")

    print(f"\n--- Done ---")
    print(f"Successfully processed: {count_success} of {len(image_files)}")

if __name__ == "__main__":
    main()