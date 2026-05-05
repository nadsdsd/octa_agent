import cv2
import math
import numpy as np

FAZ300_SIM_ANNULUS_WIDTH = 15

def largest_component(mask_bin: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), 4, cv2.CV_32S)
    if num_labels <= 1:
        return mask_bin
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == max_label)

def calculate_metrics_from_masks(pred_faz_mask: np.ndarray, pred_rv_mask: np.ndarray) -> dict:
    """接收模型输出的二值化 numpy 掩码 (224x224)，计算所有指标"""
    metrics = {}
    
    # --- 1. FAZ 指标计算 ---
    faz_clean = largest_component(pred_faz_mask)
    faz_area = int(faz_clean.sum())
    
    contours, _ = cv2.findContours(faz_clean.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    faz_perim = float(cv2.arcLength(contours[0], True)) if contours else 0.0
    faz_circularity = (4.0 * math.pi * faz_area / (faz_perim ** 2)) if faz_perim > 0 else 0.0
    
    metrics.update({
        "faz_area_px": faz_area,
        "faz_perim_px": round(faz_perim, 2),
        "faz_circularity": round(faz_circularity, 4)
    })
    
    # --- 2. RV (视网膜血管) 指标计算 ---
    total_pixels = pred_rv_mask.size
    rv_flow_area = int(pred_rv_mask.sum())
    rv_density = rv_flow_area / total_pixels if total_pixels > 0 else 0.0
    
    # 骨架化
    skel = np.zeros(pred_rv_mask.shape, np.uint8)
    img = pred_rv_mask.astype(np.uint8) * 255
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
            
    skel_bool = skel > 0
    rv_line_density = float(skel_bool.sum()) / total_pixels if total_pixels > 0 else 0.0
    
    # 分支点
    kernel = np.ones((3, 3), dtype=np.uint8)
    neigh_sum = cv2.filter2D(skel_bool.astype(np.uint8), -1, kernel)
    branch_points = int((skel_bool & (neigh_sum > 3)).sum())
    
    metrics.update({
        "rv_density": round(rv_density, 4),
        "rv_flow_area_px": rv_flow_area,
        "rv_line_density_px-1": round(rv_line_density, 4),
        "rv_branch_points": branch_points
    })
    
    # --- 3. 模拟 FAZ300 密度 ---
    k_size = FAZ300_SIM_ANNULUS_WIDTH * 2 + 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_faz = cv2.dilate(faz_clean.astype(np.uint8), dilate_kernel).astype(bool)
    annulus_mask = dilated_faz & ~faz_clean
    annulus_area = annulus_mask.sum()
    
    if annulus_area > 0:
        vessels_in_annulus = pred_rv_mask & annulus_mask
        faz300_density = float(vessels_in_annulus.sum()) / float(annulus_area)
    else:
        faz300_density = 0.0
        
    metrics["faz300_sim_density"] = round(faz300_density, 4)
    
    return metrics