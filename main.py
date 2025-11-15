import cv2
import numpy as np
from ultralytics import YOLO
from math import sqrt
import os


MODEL_PATH = 'model/best.pt'
CLASS_NAMES = {
    0: 'calcada', 1: 'carro', 2: 'faixa_pedestre', 3: 'guia_amarela',
    4: 'guia_normal', 5: 'guia_rebaixada', 6: 'placa_proibido', 7: 'rampa', 8: 'rua'
}
VEHICLE_CLASS_NAME = 'carro'

PROHIBITED_INTERSECTION_CLASSES = [
    'calcada', 'faixa_pedestre', 'guia_amarela', 'guia_rebaixada', 'rampa'
]

PROHIBITED_RELATIONAL_CLASS = 'placa_proibido'

INTERSECTION_THRESHOLDS = {
    'calcada': 50,
    'faixa_pedestre': 15,
    'guia_amarela': 50,
    'guia_rebaixada': 50,
    'rampa': 50
}

DYNAMIC_DISTANCE_FACTOR = 2.0
MIN_AREA_RATIO_CAR_TO_PLACA = 5.0
MAX_AREA_RATIO_CAR_TO_PLACA = 100.0

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_box_area(box):
    x1, y1, x2, y2 = box
    return (x2 - x1) * (y2 - y1)

def get_width(box):
    x1, y1, x2, y2 = box
    return x2 - x1

def is_near(box_carro, box_placa, max_distance):
    centro_carro = get_center(box_carro)
    centro_placa = get_center(box_placa)
    distancia = sqrt((centro_carro[0] - centro_placa[0])**2 + (centro_carro[1] - centro_placa[1])**2)
    return distancia < max_distance, distancia

def get_violation_details(infringing_class):
    """Retorna o tipo e a intensidade da infração."""
    if infringing_class == 'calcada':
        return "Estacionado na Calcada", "Grave"
    elif infringing_class == 'faixa_pedestre':
        return "Estacionado na Faixa", "Grave"
    elif infringing_class == 'guia_rebaixada':
        return "Obstruindo Garagem", "Média"
    elif infringing_class == 'guia_amarela':
        return "Local Proibido (Guia Amarela)", "Média"
    elif infringing_class == 'rampa':
        return "Estacionado na Rampa", "Grave"
    elif infringing_class == 'placa_proibido':
        return "Estacionado sob Placa Proibida", "Grave"
    else:
        return f"Estacionado em {infringing_class}", "Grave"

def get_unique_filename(output_folder, base_name, ext):
    """
    Gera um nome de arquivo único checando se ele já existe.
    Ex: 'detec_calcada.jpg', 'detec_calcada1.jpg', 'detec_calcada2.jpg'
    """
    base_path = os.path.join(output_folder, f"{base_name}{ext}")
    
    if not os.path.exists(base_path):
        return base_path
    
    counter = 1
    while True:
        new_name = f"{base_name}{counter}{ext}"
        new_path = os.path.join(output_folder, new_name)
        
        if not os.path.exists(new_path):
            return new_path
        
        counter += 1

#Processamento Principal
def process_frame_segmentation(frame, model):
    
    results = model(frame, verbose=False)[0]
    
    target_height, target_width = frame.shape[:2]

    # Listas para armazenar os diferentes tipos de detecções
    car_detections = []
    prohibited_zone_masks = {}  
    placa_detections = []
    
    detected_class_names = set()
    display_frame = frame.copy()

    # 1. Extrai e separa as máscaras e boxes
    if results.masks is None:
        print("Nenhum objeto (máscara) detectado pelo modelo.")
        if not any(CLASS_NAMES.get(int(box.cls[0]), 'desconhecido') == VEHICLE_CLASS_NAME for box in results.boxes):
             return display_frame, detected_class_names, "NONECAR"
        
    for i, mask_tensor in enumerate(results.masks.data):
        class_id = int(results.boxes[i].cls[0])
        class_name = CLASS_NAMES.get(class_id, 'desconhecido')
        
        detected_class_names.add(class_name)
        
        mask_np_uint8 = mask_tensor.cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(
            mask_np_uint8, 
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST
        )
        mask_np_bool = mask_resized.astype(bool)
        
        bbox = results.boxes[i].xyxy[0].cpu().numpy().astype(int)

        if class_name == VEHICLE_CLASS_NAME:
            car_detections.append({'bbox': bbox, 'mask': mask_np_bool})
        elif class_name in PROHIBITED_INTERSECTION_CLASSES:
            if class_name not in prohibited_zone_masks:
                prohibited_zone_masks[class_name] = []
            prohibited_zone_masks[class_name].append(mask_np_bool)
        elif class_name == PROHIBITED_RELATIONAL_CLASS:
            placa_detections.append({'bbox': bbox, 'mask': mask_np_bool})
    
    print("\n--- Análise de Veículos ---")
    if not car_detections:
        print("Nenhum carro detectado na imagem.")
        return display_frame, detected_class_names, "NONECAR"


    has_placas_in_scene = len(placa_detections) > 0
    
    #FASE 1: ANÁLISE (Sem desenhar)
    processed_cars = []

    # 2. Verifica infrações para CADA carro
    for i, car_data in enumerate(car_detections):
        car_id = i + 1
        car_bbox = car_data['bbox']
        car_mask = car_data['mask']
        car_area = get_box_area(car_bbox)
        
        infractions_for_this_car = []
        is_placa_infraction = False 
        
        #LÓGICA 2: INFRAÇÃO RELACIONAL (Placa Proibida)
        if has_placas_in_scene:
            car_box_area = car_area 
            car_width = get_width(car_bbox)
            
            for placa_data in placa_detections:
                placa_box = placa_data['bbox']
                placa_box_area = get_box_area(placa_box)
                if placa_box_area < 1: 
                    continue 

                area_ratio = car_box_area / placa_box_area
                
                if not (MIN_AREA_RATIO_CAR_TO_PLACA <= area_ratio <= MAX_AREA_RATIO_CAR_TO_PLACA):
                    continue 

                max_allowed_distance = car_width * DYNAMIC_DISTANCE_FACTOR
                esta_perto, dist = is_near(car_bbox, placa_box, max_allowed_distance)
                
                if esta_perto: 
                    tipo, intensidade = get_violation_details(PROHIBITED_RELATIONAL_CLASS)
                    infractions_for_this_car.append({
                        "class_name": PROHIBITED_RELATIONAL_CLASS, 
                        "tipo": tipo,
                        "intensidade": intensidade,
                        "detalhe": f"Proximo a placa (Ratio: {area_ratio:.1f}, Dist: {dist:.0f}px)"
                    })
                    is_placa_infraction = True 
                    break 
                
        #LÓGICA 1: INFRAÇÃO POR INTERSEÇÃO (Calçada, Faixa, etc.)
        if not is_placa_infraction:
            is_faixa_infraction = False 
            
            #LÓGICA 1.1: Prioridade Faixa de Pedestre
            if 'faixa_pedestre' in prohibited_zone_masks:
                class_name = 'faixa_pedestre'
                threshold = INTERSECTION_THRESHOLDS.get(class_name, 50)
                
                combined_mask_for_class = np.zeros_like(car_mask, dtype=bool)
                for p_mask in prohibited_zone_masks[class_name]:
                    combined_mask_for_class = np.logical_or(combined_mask_for_class, p_mask)
                
                current_intersection = np.logical_and(car_mask, combined_mask_for_class)
                current_pixels = np.sum(current_intersection)
                
                if current_pixels > threshold:
                    is_faixa_infraction = True 
                    tipo, intensidade = get_violation_details(class_name)
                    infractions_for_this_car.append({
                        "class_name": class_name,
                        "tipo": tipo,
                        "intensidade": intensidade,
                        "detalhe": f"Sobreposicao: {current_pixels} pixels em '{class_name}'"
                    })

            #LÓGICA 1.2: Demais Infrações (Calçada, Guias, etc.)
            if not is_faixa_infraction:
                max_pixels_intersection = 0
                infringing_class_intersection = None

                for class_name, masks_list in prohibited_zone_masks.items():
                    if class_name == 'faixa_pedestre':
                        continue
                    threshold = INTERSECTION_THRESHOLDS.get(class_name, 50) 
                    combined_mask_for_class = np.zeros_like(car_mask, dtype=bool)
                    for p_mask in masks_list:
                        combined_mask_for_class = np.logical_or(combined_mask_for_class, p_mask)
                    
                    current_intersection = np.logical_and(car_mask, combined_mask_for_class)
                    current_pixels = np.sum(current_intersection)
                    
                    if current_pixels > threshold:
                        if current_pixels > max_pixels_intersection:
                            max_pixels_intersection = current_pixels
                            infringing_class_intersection = class_name
                
                if infringing_class_intersection:
                    tipo, intensidade = get_violation_details(infringing_class_intersection)
                    infractions_for_this_car.append({
                        "class_name": infringing_class_intersection,
                        "tipo": tipo,
                        "intensidade": intensidade,
                        "detalhe": f"Sobreposicao: {max_pixels_intersection} pixels em '{infringing_class_intersection}'"
                    })
        
        if infractions_for_this_car:
            print(f"  [CARRO {car_id}] - {len(infractions_for_this_car)} INFRAÇÃO(ÕES) DETECTADA(S):")
            for infr in infractions_for_this_car:
                print(f"    - {infr['tipo']} ({infr['intensidade']}). Detalhe: {infr['detalhe']}")
        else:
            print(f"  [CARRO {car_id}] - OK. Estacionamento regular.")
            
        processed_cars.append({
            "car_id": car_id,
            "bbox": car_bbox,
            "mask": car_mask,
            "area": car_area,
            "infractions": infractions_for_this_car
        })
        

    #FASE 2: DESENHO
    
    #Encontra o infrator principal (maior área)
    primary_violator_id = None
    violators = [car for car in processed_cars if car["infractions"]]
    
    if violators:
        primary_violator = max(violators, key=lambda car: car['area'])
        primary_violator_id = primary_violator['car_id']

    for car in processed_cars:
        car_bbox = car['bbox']
        car_mask = car['mask']
        is_primary = (car['car_id'] == primary_violator_id)
        is_violator = bool(car['infractions'])
        
        label = ""
        label_detail = ""
        
        if is_primary:
            color = (0, 0, 255) # Vermelho (Principal)
        elif is_violator:
            color = (0, 255, 255) # Amarelo (Secundário)
        else:
            color = (0, 255, 0) # Verde (OK)
            
        if is_violator:
            primeira_infracao = car['infractions'][0]
            label = f"INFRA: {primeira_infracao['tipo']} ({primeira_infracao['intensidade']})"
            label_detail = primeira_infracao['detalhe']
            if len(car['infractions']) > 1:
                label += " (+...)"
        else:
            label = "OK"

        if is_violator:
            overlay = display_frame.copy()
            overlay[car_mask] = overlay[car_mask] * 0.5 + np.array(color) * 0.5
            display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
        
        cv2.rectangle(display_frame, (car_bbox[0], car_bbox[1]), (car_bbox[2], car_bbox[3]), color, 2)
        cv2.putText(display_frame, label, (car_bbox[0], car_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if label_detail:
             cv2.putText(display_frame, label_detail, (car_bbox[0], car_bbox[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 5. Desenha zonas proibidas (INTERSEÇÃO)
    for class_name, masks_list in prohibited_zone_masks.items():
        for p_mask in masks_list:
            overlay = display_frame.copy()
            if class_name == 'calcada': zone_color = (0, 255, 255) # Amarelo
            elif class_name == 'faixa_pedestre': zone_color = (255, 0, 255) # Magenta
            elif class_name == 'guia_amarela': zone_color = (0, 165, 255) # Laranja
            elif class_name == 'guia_rebaixada': zone_color = (255, 255, 0) # Ciano
            elif class_name == 'rampa': zone_color = (128, 0, 128) # Roxo
            else: zone_color = (255, 0, 0) # Azul
            
            overlay[p_mask] = overlay[p_mask] * 0.5 + np.array(zone_color) * 0.5
            display_frame = cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0)

    # 6. Desenha zonas proibidas (RELACIONAL - PLACAS)
    zone_color_placa = (255, 0, 255) # Magenta (mesma da faixa, mas é só um sinal)
    for placa_data in placa_detections:
        p_mask = placa_data['mask']
        overlay = display_frame.copy()
        overlay[p_mask] = overlay[p_mask] * 0.5 + np.array(zone_color_placa) * 0.5
        display_frame = cv2.addWeighted(overlay, 0.4, display_frame, 0.6, 0)

    status_key = "OK"
    if primary_violator_id:
        primary_violator = next(car for car in processed_cars if car['car_id'] == primary_violator_id)
        status_key = primary_violator['infractions'][0]['class_name']
    
    return display_frame, detected_class_names, status_key

def main():
    try:
        model = YOLO(MODEL_PATH)
        print(f"Modelo {MODEL_PATH} (segmentação) carregado com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return
    
    SOURCE_IMAGE_PATH = 'imagens/image14.png' 
    OUTPUT_FOLDER = 'resultadoUnico'
    
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        print(f"Pasta de resultados '{OUTPUT_FOLDER}' pronta.")
    except Exception as e:
        print(f"Erro ao criar pasta de resultados: {e}")
        return
        
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    
    try:
        filename = os.path.basename(SOURCE_IMAGE_PATH)
        file_ext = os.path.splitext(filename)[1].lower()
    except Exception as e:
        print(f"Erro ao processar o caminho da imagem '{SOURCE_IMAGE_PATH}': {e}")
        return

    if file_ext not in allowed_extensions:
        print(f"Erro: Arquivo '{filename}' não é uma imagem suportada (use .jpg, .jpeg, .png).")
        return

    try:
        frame = cv2.imread(SOURCE_IMAGE_PATH)
        if frame is None:
            if not os.path.exists(SOURCE_IMAGE_PATH):
                 raise FileNotFoundError(f"Arquivo não encontrado: {SOURCE_IMAGE_PATH}")
            else:
                 raise Exception(f"Não foi possível ler a imagem: {SOURCE_IMAGE_PATH}. O arquivo pode estar corrompido.")
            
        print(f"\nProcessando imagem: {SOURCE_IMAGE_PATH}...")
        
        result_frame, all_elements, status_key = process_frame_segmentation(frame.copy(), model)
        
        base_name = f"detec_{status_key}"
        
        output_image_path = get_unique_filename(OUTPUT_FOLDER, base_name, file_ext)
        
        cv2.imwrite(output_image_path, result_frame)
        print(f"Imagem '{output_image_path}' salva com sucesso.")
        
        print("\n--- Relatório de Elementos ---")
        if all_elements:
            element_list = ", ".join(sorted(list(all_elements)))
            print(f"Todos os elementos detectados na imagem: {element_list}")
        else:
            print("Nenhum elemento foi detectado na imagem.")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        import traceback
        print(f"Erro ao processar imagem {filename}:")
        print(traceback.format_exc())

    print("\nProcessamento da imagem única concluído.")


if __name__ == "__main__":
    main()