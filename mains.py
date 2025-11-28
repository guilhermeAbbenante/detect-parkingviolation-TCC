import cv2
import numpy as np
from ultralytics import YOLO
from math import sqrt
import os
from typing import List, Dict, Tuple, Set, Any

# ==============================================================================
# CONFIGURAÇÃO GERAL
# ==============================================================================
MODEL_PATH = 'model/best.pt'
INPUT_FOLDER = 'imagens'
OUTPUT_FOLDER = 'resultados'

CLASS_NAMES = {
    0: 'calcada', 1: 'carro', 2: 'faixa_pedestre', 3: 'guia_amarela',
    4: 'guia_normal', 5: 'guia_rebaixada', 6: 'placa_proibido', 7: 'rampa', 8: 'rua'
}

VEHICLE_CLASS = 'carro'
RELATIONAL_CLASS = 'placa_proibido'

PROHIBITED_ZONES = [
    'calcada', 'faixa_pedestre'
]

#CONFIGURAÇÕES ESPECÍFICAS DE MÁSCARAS

# Configuração para FAIXA DE PEDESTRE
MASK_PERCENTAGE_FAIXA = 0.60  

# Configuração para CALÇADA
MASK_PERCENTAGE_CALCADA = 0.15 
DILATION_KERNEL_SIZE = 15     

# Limiares de Detecção
THRESHOLDS = {
    'calcada': 60,
    'faixa_pedestre': 15,
    'outros': 50
}

# Configurações de Placas
DYNAMIC_DISTANCE_FACTOR = 2.0
MIN_RATIO_CAR_PLACA = 5.0
MAX_RATIO_CAR_PLACA = 100.0

# ==============================================================================
# FUNÇÕES UTILITÁRIAS
# ==============================================================================

def get_bottom_mask(full_mask: np.ndarray, percentage: float) -> np.ndarray:
    """Retorna apenas a porcentagem inferior da máscara."""
    rows = np.any(full_mask, axis=1)
    if not np.any(rows): return full_mask
    
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
        'placa_proibido': ("Estacionado sob Placa Proibida", "Grave")
    }
    return data.get(class_name, (f"Estacionado em {class_name}", "Grave"))

def get_unique_filename(folder: str, base_name: str, ext: str) -> str:
    """
    Gera um nome único sequencial. 
    Ex: calcada1.jpg, calcada2.jpg, etc.
    """
    counter = 1
    while True:
        path = os.path.join(folder, f"{base_name}{counter}{ext}")
        if not os.path.exists(path):
            return path
        counter += 1

# ==============================================================================
# ANALISE DE INFRAÇÕES
# ==============================================================================

def parse_detections(results, w: int, h: int) -> Dict:
    data = {'cars': [], 'zones': {}, 'plates': [], 'detected_classes': set()}

    if results.masks is None: return data

    for i, mask_tensor in enumerate(results.masks.data):
        class_id = int(results.boxes[i].cls[0])
        class_name = CLASS_NAMES.get(class_id, 'desconhecido')
        data['detected_classes'].add(class_name)
        
        mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        bbox = results.boxes[i].xyxy[0].cpu().numpy().astype(int)
        obj_data = {'bbox': bbox, 'mask': mask_resized}

        if class_name == VEHICLE_CLASS: data['cars'].append(obj_data)
        elif class_name in PROHIBITED_ZONES:
            if class_name not in data['zones']: data['zones'][class_name] = []
            data['zones'][class_name].append(mask_resized)
        elif class_name == RELATIONAL_CLASS: data['plates'].append(obj_data)
    
    return data

def check_plate_violations(car: Dict, plates: List[Dict]) -> bool:
    if not plates: return False
    car_w = get_width(car['bbox'])
    for plate in plates:
        plate_area = get_box_area(plate['bbox'])
        if plate_area < 1: continue
        ratio = car['area'] / plate_area
        if not (MIN_RATIO_CAR_PLACA <= ratio <= MAX_RATIO_CAR_PLACA): continue
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
    mask_faixa_pedestre = get_bottom_mask(car['mask'], MASK_PERCENTAGE_FAIXA)
    
    mask_base_calcada = get_bottom_mask(car['mask'], MASK_PERCENTAGE_CALCADA).astype(np.uint8)
    kernel = np.ones((DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE), np.uint8)
    mask_calcada_dilated = cv2.dilate(mask_base_calcada, kernel, iterations=1).astype(bool)
    area_dilated_calcada = np.count_nonzero(mask_calcada_dilated)

    best_violation = None
    
    for cls_name, masks in zones.items():
        combined_zone = np.zeros_like(car['mask'], dtype=bool)
        for m in masks: combined_zone = np.logical_or(combined_zone, m)

        violation_detected = False
        detalhe = ""
        prioridade = 0 

        if cls_name == 'faixa_pedestre':
            overlap = np.sum(np.logical_and(mask_faixa_pedestre, combined_zone))
            if overlap > THRESHOLDS['faixa_pedestre']:
                violation_detected = True
                prioridade = 3
                detalhe = f"Faixa: {overlap}px"

        elif cls_name == 'calcada':
            if area_dilated_calcada > 0:
                overlap = np.sum(np.logical_and(mask_calcada_dilated, combined_zone))
                if overlap > THRESHOLDS['calcada']:
                    violation_detected = True
                    prioridade = 2  # ALTERADO: Diminuí para 2 (Menor que faixa)
                    detalhe = f"Calcada: {overlap}px"

        if violation_detected:
            if best_violation is None or prioridade > best_violation['prio']:
                best_violation = {'class_name': cls_name, 'detalhe': detalhe, 'prio': prioridade}

    if best_violation:
        tipo, grav = get_violation_text(best_violation['class_name'])
        car['infractions'].append({
            'class_name': best_violation['class_name'],
            'tipo': tipo,
            'intensidade': grav,
            'detalhe': best_violation['detalhe']
        })

def analyze_infractions(data: Dict) -> List[Dict]:
    processed_cars = []
    for i, car in enumerate(data['cars']):
        car_info = {
            'id': i + 1, 'bbox': car['bbox'], 'mask': car['mask'],
            'area': get_box_area(car['bbox']), 'infractions': []
        }
        if not check_plate_violations(car_info, data['plates']):
            check_ground_violations(car_info, data['zones'])
            
        processed_cars.append(car_info)
    return processed_cars

# ==============================================================================
# VISUALIZAÇÃO
# ==============================================================================

def draw_visuals(frame: np.ndarray, cars: List[Dict], zones: Dict, plates: List[Dict]) -> Tuple[np.ndarray, str]:
    if not cars:
        return frame, "NONECAR"

    primary_car = max(cars, key=lambda c: c['area'])
    primary_id = primary_car['id']
    
    if primary_car['infractions']:
        status_key = primary_car['infractions'][0]['class_name']
    else:
        status_key = "OK"

    for car in cars:
        is_primary = (car['id'] == primary_id)
        is_violator = bool(car['infractions'])
        
        if is_primary:
            color = (0, 0, 255) if is_violator else (0, 255, 0)
        else:
            color = (0, 255, 255) if is_violator else (0, 255, 0)

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
# MAIN (ALTERADO PARA LOTE)
# ==============================================================================

def process_image(frame: np.ndarray, model: YOLO) -> Tuple[np.ndarray, Set[str], str]:
    h, w = frame.shape[:2]
    display_frame = frame.copy()
    
    results = model(frame, verbose=False)[0]
    parsed_data = parse_detections(results, w, h)
    
    if not parsed_data['cars']:
        return display_frame, parsed_data['detected_classes'], "NONECAR"

    processed_cars = analyze_infractions(parsed_data)
    final_frame, status_key = draw_visuals(display_frame, processed_cars, parsed_data['zones'], parsed_data['plates'])

    return final_frame, parsed_data['detected_classes'], status_key

def main():
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Carregando modelo: {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Modelo carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    if not os.path.exists(INPUT_FOLDER):
        print(f"Erro: Pasta de entrada '{INPUT_FOLDER}' não encontrada.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    print(f"\nEncontradas {len(image_files)} imagens para processar na pasta '{INPUT_FOLDER}'.")

    for i, filename in enumerate(image_files):
        source_path = os.path.join(INPUT_FOLDER, filename)
        
        try:
            print(f"[{i+1}/{len(image_files)}] Processando: {filename}...")
            frame = cv2.imread(source_path)
            if frame is None:
                print(f"  -> Erro: Falha ao ler imagem.")
                continue

            result_frame, classes, status_key = process_image(frame, model)
            
            if status_key == "NONECAR":
                base_name = "NONE"
            else:
                base_name = status_key

            ext = os.path.splitext(filename)[1]
            out_path = get_unique_filename(OUTPUT_FOLDER, base_name, ext)
            
            cv2.imwrite(out_path, result_frame)
            print(f"  -> Salvo como: {os.path.basename(out_path)}")

        except Exception as e:
            import traceback
            print(f"  -> Erro no processamento de {filename}:\n{traceback.format_exc()}")

    print("\n--- Processamento em lote finalizado ---")

if __name__ == "__main__":
    main()