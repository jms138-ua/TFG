import numpy as np
import torch
import scipy
import cv2
import ot
import trimesh
import matplotlib as mpl
import matplotlib.pyplot as plt

import os
import time
import itertools
import json
import xml
import csv

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

from densepose import add_densepose_config

from libs.STAR.star.pytorch.star import STAR

#==================================================================================================


#Evitar warnings a mitad de ejecucion
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.meshgrid:.*indexing argument")
warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning, message="Unknown solver options:*")
torch.nn.Conv2d(3, 16, 3)(torch.randn(1, 3, 64, 64))


#==================================================================================================


DATASETPATH = "data/testJorge4x6"
RESULTPATH = "results/main-temp/"
os.makedirs(RESULTPATH, exist_ok=True)


#==================================================================================================


DP_BODY_PARTS = {
    1: "Torso", 2: "Torso",
    3: "Right Hand", 4: "Left Hand",
    5: "Left Foot", 6: "Right Foot",
    7: "Upper Leg Right", 8: "Upper Leg Left",
    9: "Upper Leg Right", 10: "Upper Leg Left",
    11: "Lower Leg Right", 12: "Lower Leg Left",
    13: "Lower Leg Right", 14: "Lower Leg Left",
    15: "Upper Arm Left", 16: "Upper Arm Right",
    17: "Upper Arm Left", 18: "Upper Arm Right",
    19: "Lower Arm Left", 20: "Lower Arm Right",
    21: "Lower Arm Left", 22: "Lower Arm Right",
    23: "Head", 24: "Head"
}

NTHETAS = 72 #number of pose parameters -> 24 joints with 3 rotation values ​​each
NBETAS = 10 #number of shape parameters
NTRANS = 3 #number of translation paramete

POSE_BOUNDS = np.array([(-np.pi,np.pi)]*NTHETAS)
SHAPE_BOUNDS = np.array([(-2.0,2.0)]*NBETAS)

BODY_PART_TO_SMPL_POSE_INDICES = {
    "Torso": list(range(0, 3)) + list(range(9, 12)) + list(range(18, 21)) + list(range(27, 30)),
    "Upper Leg Left": list(range(3, 6)),
    "Upper Leg Right": list(range(6, 9)),
    "Lower Leg Left": list(range(12, 15)),
    "Lower Leg Right": list(range(15, 18)),
    "Left Foot": list(range(21, 24)) + list(range(30, 33)),
    "Right Foot": list(range(24, 27)) + list(range(33, 36)),
    "Head": list(range(36, 39)) + list(range(45, 48)),
    "Upper Arm Left": list(range(39, 42)) + list(range(48, 51)),
    "Upper Arm Right": list(range(42, 45)) + list(range(51, 54)),
    "Lower Arm Left": list(range(54, 57)) + list(range(60, 63)),
    "Lower Arm Right": list(range(57, 60)) + list(range(63, 66)),
    "Left Hand": list(range(66, 69)),
    "Right Hand": list(range(69, 72)),
}

DP_TO_SMPL = {
    "Torso": ["spine1", "spine2", "spine", "hips"],
    "Right Hand": ["rightHand", "rightHandIndex1"],
    "Left Hand": ["leftHand", "leftHandIndex1"],
    "Left Foot": ["leftFoot", "leftToeBase"],
    "Right Foot": ["rightFoot", "rightToeBase"],
    "Upper Leg Right": ["rightUpLeg"],
    "Upper Leg Left": ["leftUpLeg"],
    "Lower Leg Right": ["rightLeg"],
    "Lower Leg Left": ["leftLeg"],
    "Upper Arm Right": ["rightArm", "rightShoulder"],
    "Upper Arm Left": ["leftArm", "leftShoulder"],
    "Lower Arm Right": ["rightForeArm"],
    "Lower Arm Left": ["leftForeArm"],
    "Head": ["head", "neck"],
}

CMAP = "tab20"

#==================================================================================================


def load_densepose():
    dp_cfg_iuv = get_cfg()
    dp_cfg_iuv.MODEL.DEVICE = "cpu"
    add_densepose_config(dp_cfg_iuv)
    dp_cfg_iuv.merge_from_file("libs/detectron2/projects/DensePose/configs/densepose_rcnn_R_101_FPN_DL_WC1M_s1x.yaml")
    dp_cfg_iuv.MODEL.WEIGHTS = "libs/detectron2/projects/DensePose/models/model_final_0ebeb3.pkl"
    dp_predictor_iuv = DefaultPredictor(dp_cfg_iuv)

    return dp_predictor_iuv


def load_scans():
    SCANPATH = DATASETPATH+"/"

    with open(DATASETPATH+".csv") as csvf:
        scans = [[row["SCANID"], int(row["ROTATIONZ"])]
            for row in csv.DictReader(csvf)
        ]

    for scanid, rotationz in scans:
        image = cv2.imread(f"{SCANPATH}{scanid}-Color-1-calibrated.png")

        mesh = trimesh.load_mesh(f"{SCANPATH}{scanid}-MaskPointCloud-0.ply", file_type="ply", process=False)
        vmesh_scan = mesh.vertices

        cameraf = xml.etree.ElementTree.parse(f"{SCANPATH}cameras.xml")
        calib = {}
        for sensor in cameraf.getroot().findall(".//sensor"):
            if sensor.attrib.get("id") == scanid:
                calibration = sensor.find("calibration")
                for param in calibration:
                    if param.tag in ("fx", "fy", "cx", "cy"):
                        calib[param.tag] = float(param.text)
        K = np.array([
            [calib["fx"], 0, calib["cx"]],
            [0, calib["fy"], calib["cy"]],
            [0,           0,           1]
        ])

        if rotationz != 0:
            h, w = image.shape[:2]
            center = (w/2, h/2)

            M = cv2.getRotationMatrix2D(center=center, angle=-rotationz, scale=1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int(h*sin + w*cos)
            new_h = int(h*cos + w*sin)

            M[0,2] += (new_w-w) / 2
            M[1,2] += (new_h-h) / 2

            image = cv2.warpAffine(image, M, (new_w, new_h))
            M_homogeneous = np.vstack([M, [0, 0, 1]])
            K = M_homogeneous @ K

        yield scanid, image, vmesh_scan, K


def load_star():
    def getAPose(thetas):
        deg2rad = np.pi / 180
        Apose = np.zeros(thetas)
        Apose[48:51] = np.multiply(deg2rad, [0,0,-70]) #Left shoulder
        Apose[51:54] = np.multiply(deg2rad, [0,0,70]) #Right shoulder
        Apose[54:57] = np.multiply(deg2rad, [0,-20,0]) #Left elbow
        Apose[57:60] = np.multiply(deg2rad, [0,20,0]) #Right elbow
        return Apose

    with open("data/smpl_vert_segmentation.json", "r") as f:
        smpl_vert_segmentation = json.load(f)

    star = STAR(gender="neutral", num_betas=NBETAS)
    pose = torch.tensor(getAPose(NTHETAS), dtype=torch.float32).unsqueeze(0)
    shape = torch.zeros(1, NBETAS)
    trans = torch.zeros(1, NTRANS)
    star_model = star(pose, shape, trans)
    vstar_model = f_vstar(star_model)

    return star, (pose, shape, trans), star_model, vstar_model, smpl_vert_segmentation


def load_synthetic_star(star, starparams):
    pose, shape, trans = starparams

    synt_shape = torch.tensor([
        0.0, #linked to height
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0,
        2.0
    ], dtype=torch.float32).unsqueeze(0)

    synt_star_model = star(pose, synt_shape, trans)
    synt_vstar_model = f_vstar(synt_star_model)

    return synt_star_model, synt_vstar_model


#==================================================================================================


def extract_submesh_faces(faces_full, vids):
    idx_map = {v: i for i, v in enumerate(np.array(vids))}
    return np.array([
        [idx_map[v] for v in face]
            for face in faces_full
                if all(v in idx_map for v in face)
    ])


def fixed_coloring_mask(segm_mask):
    '''
    Converts a segmentation mask to a consistent part index mask
    '''

    labels = sorted(set(DP_BODY_PARTS.values()))

    cmap = mpl.colormaps.get_cmap(CMAP)

    fixed_segm_mask = np.vectorize(
        lambda pid: labels.index(DP_BODY_PARTS[pid])+1 if pid in DP_BODY_PARTS else 0
    )(segm_mask)
    colored_mask = cmap(np.ma.masked_where(fixed_segm_mask == 0, fixed_segm_mask))

    return colored_mask


def fixed_coloring_parts(dict_vids):
    '''
    Converts a vertex-to-part mapping to consistent colors per part
    '''
    labels = sorted(set(DP_BODY_PARTS.values()))

    cmap = mpl.colormaps.get_cmap(CMAP)

    fixed_pids = [
        labels.index(dict_vids[vid])+1
            for vid in dict_vids.keys()
    ]
    vertices_color = cmap(fixed_pids)

    return vertices_color


#==================================================================================================


def smooth_labels(segm_mask, min_component_size=81, structure_size=(3,3)):
    '''
    Reduces noisy parts
    '''

    smooth_segm_mask = np.copy(segm_mask)

    pixels_to_relabel = np.zeros_like(segm_mask, dtype=bool)
    for pid in np.unique(smooth_segm_mask):
        if pid not in DP_BODY_PARTS:
            continue

        part_mask = (smooth_segm_mask == pid)

        #Small components as background
        components_map_mask, num_components = scipy.ndimage.label(
            part_mask
        )
        for i in range(1, num_components+1):
            component_mask = (components_map_mask == i)
            if np.sum(component_mask) < min_component_size:
                pixels_to_relabel[component_mask] = True
                smooth_segm_mask[component_mask] = 0

    #Fills unlabeled pixels
    distances, (nearest_y, nearest_x) = scipy.ndimage.distance_transform_edt(
        smooth_segm_mask == 0,
        return_indices=True
    )
    smooth_segm_mask[pixels_to_relabel] = smooth_segm_mask[nearest_y[pixels_to_relabel], nearest_x[pixels_to_relabel]]

    #Morphological opening (erosion then dilation) to remove small noise and smooth object contours
    structure = np.ones(structure_size, dtype=bool)
    smooth_segm_mask = scipy.ndimage.binary_opening(
        smooth_segm_mask,
        structure=structure
    ).astype(smooth_segm_mask.dtype) * smooth_segm_mask

    return smooth_segm_mask


def correct_anatomical_anomalies(segm_mask, min_pixels_for_anomaly=100, proximity_threshold_px=150):
    """
    Corrects anatomical anomalies by facing extremes of body height
    """

    ROLES = {
        "Hand", "Foot", "Upper Arm", "Lower Arm",
        "Upper Leg", "Lower Leg", "Torso", "Head"
    }
    ROLES_UPPER_BODY = {"Hand", "Lower Arm", "Upper Arm", "Head"}
    ROLES_LOWER_BODY = {"Foot", "Lower Leg", "Upper Leg"}

    ANOMALOUS_PROXIMITY_PAIRS = {
        frozenset({"Hand", "Foot"}),
        frozenset({"Hand", "Lower Leg"}),
        frozenset({"Lower Arm", "Foot"}),
        frozenset({"Lower Arm", "Lower Leg"}),
        frozenset({"Upper Arm", "Foot"}),
        frozenset({"Upper Arm", "Lower Leg"}),
        frozenset({"Upper Arm", "Upper Leg"}),
        frozenset({"Head", "Foot"}),
        frozenset({"Head", "Lower Leg"}),
        frozenset({"Head", "Upper Leg"}),
    }

    correct_segm_mask = np.copy(segm_mask)

    #Extract information from each part
    pids_info = {}
    for pid in np.unique(correct_segm_mask):
        if pid not in DP_BODY_PARTS:
            continue

        part_mask = (correct_segm_mask == pid)

        part_size = np.sum(part_mask)
        if part_size < min_pixels_for_anomaly:
            continue

        part_name = DP_BODY_PARTS[pid]

        for role_keyword in ROLES:
            if role_keyword in part_name:
                part_role = role_keyword
                break

        y, x = scipy.ndimage.center_of_mass(part_mask)

        pids_info[pid] = {
            "size": part_size,
            "name": part_name,
            "role": part_role,
            "com_y": y,
            "com_x": x
        }

    #Check for anomalous parts
    all_person_y_coords = [part_info["com_y"] for part_info in pids_info.values()]
    estimated_body_min_y = np.min(all_person_y_coords)
    estimated_body_max_y = np.max(all_person_y_coords)
    estimated_body_height = estimated_body_max_y - estimated_body_min_y

    def is_anomalously_positioned(part_info, other_part_info, estimated_min_y, estimated_max_y, estimated_height):
        '''
        Checks if a part is anomalous based on its role and height
        '''
        threshold_y = estimated_min_y + estimated_height * 0.5

        #It is below the mean threshould Y and is higher than the other part
        if part_info["role"] in ROLES_UPPER_BODY:
            return part_info["com_y"] > threshold_y and part_info["com_y"] >= other_part_info["com_y"]
        #It is above the mean threshould Y and is lower than the other part
        elif part_info["role"] in ROLES_LOWER_BODY:
            return part_info["com_y"] < threshold_y and part_info["com_y"] <= other_part_info["com_y"]
        return False

    #Remove anomalous parts
    removed_pids = set()
    for pid_a, info_a in list(pids_info.items()):
        if pid_a in removed_pids:
            continue

        for pid_b, info_b in list(pids_info.items()):
            if pid_a in removed_pids:
                break

            if pid_b in removed_pids:
                continue

            if pid_a == pid_b:
                continue

            if frozenset({info_a["role"], info_b["role"]}) not in ANOMALOUS_PROXIMITY_PAIRS:
                continue

            distance = np.sqrt((info_a["com_y"] - info_b["com_y"])**2 + (info_a["com_x"] - info_b["com_x"])**2)
            if distance >= proximity_threshold_px:
                continue

            is_a_anomalous = is_anomalously_positioned(info_a, info_b, estimated_body_min_y, estimated_body_max_y, estimated_body_height)
            is_b_anomalous = is_anomalously_positioned(info_b, info_a, estimated_body_min_y, estimated_body_max_y, estimated_body_height)

            if is_a_anomalous and is_b_anomalous:
                #The smaller one in size
                if info_a["size"] < info_b["size"]:
                    pid_to_remove = pid_a
                else:
                    pid_to_remove = pid_b
            elif is_a_anomalous:
                pid_to_remove = pid_a
            elif is_b_anomalous:
                pid_to_remove = pid_b
            else: continue

            #Fill deleted part
            removed_part_info = pids_info[pid_to_remove]

            best_fill_pid = 0
            min_fill_distance = float("inf")
            for current_pid, current_info in pids_info.items():
                if current_pid == pid_to_remove or current_pid in removed_pids:
                    continue

                fill_distance = np.sqrt((removed_part_info["com_y"] - current_info["com_y"])**2 + (removed_part_info["com_x"] - current_info["com_x"])**2)

                is_compatible_role = False
                if removed_part_info["role"] in ROLES_UPPER_BODY and current_info["role"] in ROLES_UPPER_BODY:
                    is_compatible_role = True
                elif removed_part_info["role"] in ROLES_LOWER_BODY and current_info["role"] in ROLES_LOWER_BODY:
                    is_compatible_role = True

                if (is_compatible_role and fill_distance < min_fill_distance) or (best_fill_pid == 0 and fill_distance < proximity_threshold_px/2):
                    min_fill_distance = fill_distance
                    best_fill_pid = current_pid

            correct_segm_mask[correct_segm_mask == pid_to_remove] = best_fill_pid

            removed_pids.add(pid_to_remove)

    return correct_segm_mask


def fill_unlabeled_points(dict_segm_vids, vmesh):
    '''
    Fills unlabeled points using the nearest classified neighbor
    '''

    N = vmesh.shape[0]
    all_vids = np.arange(N)

    dict_refined_segm_vids = dict(dict_segm_vids)

    labeled_vids = np.array(list(dict_segm_vids.keys()))
    unlabeled_vids = np.setdiff1d(all_vids, labeled_vids)

    labeled_points = vmesh[labeled_vids]
    tree = scipy.spatial.cKDTree(labeled_points)
    distances, nearest = tree.query(vmesh[unlabeled_vids], k=1)
    for i, vid in enumerate(unlabeled_vids):
        neighbor_vid = labeled_vids[nearest[i]]
        dict_refined_segm_vids[vid] = dict_segm_vids[neighbor_vid]

    return dict_refined_segm_vids


#==================================================================================================


def project_points(K, points_3d):
    '''
    Porjects 3D points to 2D using the intrinsic matrix K
    '''

    points_proj_homogeneous = points_3d @ K.T
    points_2d = points_proj_homogeneous[:,:2] / points_proj_homogeneous[:,2:3]
    return np.round(points_2d).astype(int)


def align_rigid_transform(src_points, tgt_points):
    '''
    Best-fit transform that maps src_points to tgt_points
    using the Kabsch algorithm (rigid alignment via SVD)
    '''

    src_centroid = src_points.mean(axis=0)
    tgt_centroid = tgt_points.mean(axis=0)

    #Center the points
    src_centered = src_points - src_centroid
    tgt_centered = tgt_points - tgt_centroid

    #Covariance matrix
    H = src_centered.T @ tgt_centered

    #SVD
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    #Avoid reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_centroid - R @ src_centroid

    return R, t


def ransac_kabsch(src_points, tgt_points, threshold=0.02, min_inliers=3):
    '''
    Exhaustive RANSAC with the Kabsch algorithm
    obtaining the inliers, points in pairs that agree with the estimated rigid transformation
    '''

    assert src_points.shape == tgt_points.shape
    N = src_points.shape[0]
    best_inliers = []

    #Choose combination with more inliers that exceed the threshold
    for k in range(min_inliers, N+1):
        for idx in itertools.combinations(range(N), k):
            idx = list(idx)
            src_sample = src_points[idx]
            tgt_sample = tgt_points[idx]

            R_est, t_est = align_rigid_transform(src_sample, tgt_sample)
            src_aligned = (R_est @ src_points.T).T + t_est

            distances = np.linalg.norm(src_aligned - tgt_points, axis=1)
            inliers = [i for i, d in enumerate(distances) if d < threshold]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

    if len(best_inliers) >= min_inliers:
        return align_rigid_transform(src_points[best_inliers], tgt_points[best_inliers])
    else:
        return align_rigid_transform(src_points, tgt_points)


def chamfer_dist(P, Q):
    '''
    #Version con vectorizacion completa
    diff = P[:,np.newaxis,:] - Q[np.newaxis,:,:]
    dist_matrix = np.linalg.norm(diff, axis=2)
    min_dist_PQ = np.min(dist_matrix, axis=1)
    min_dist_QP = np.min(dist_matrix, axis=0)
    return np.mean(min_dist_PQ) + np.mean(min_dist_QP)
    '''
    kdtree_Q = scipy.spatial.cKDTree(Q)
    dist_P_to_Q, _ = kdtree_Q.query(P, k=1)
    kdtree_P = scipy.spatial.cKDTree(P)
    dist_Q_to_P, _ = kdtree_P.query(Q, k=1)
    return np.mean(dist_P_to_Q) + np.mean(dist_Q_to_P)


def hausdorff_dist(P, Q):
    '''
    #Version con vectorizacion completa
    diff = P[:,np.newaxis,:] - Q[np.newaxis,:,:]
    dist_matrix = np.linalg.norm(diff, axis=2)
    min_dist_PQ = np.min(dist_matrix, axis=1)
    min_dist_QP = np.min(dist_matrix, axis=0)
    return max(np.max(min_dist_PQ), np.max(min_dist_QP))
    '''
    kdtree_Q = scipy.spatial.cKDTree(Q)
    dist_P_to_Q, _ = kdtree_Q.query(P, k=1)
    kdtree_P = scipy.spatial.cKDTree(P)
    dist_Q_to_P, _ = kdtree_P.query(Q, k=1)
    return max(np.max(dist_P_to_Q), np.max(dist_Q_to_P))


#==================================================================================================


f_vstar = lambda star_model: star_model.detach().numpy()[0]

print("\n... INICIANDO ...")
t0 = time.time()
tsoptm = []
dp_predictor_iuv = load_densepose()
star, (pose, shape, trans), star_model, vstar_model, smpl_vert_segmentation = load_star()
synt_star_model, synt_vstar_model = load_synthetic_star(star, (pose, shape, trans))

print("\n==============================\n")
print("... OBTENIENDO MODELOS DE ESCANEOS PARCIALES ...")

optm_poses = []
optm_shapes = []
optm_synt_poses = []
optm_synt_shapes = []

for scanid, image, vmesh_scan, K in load_scans():
    print(f"\n=== PROCESANDO CAMARA {scanid} ===")

    #SEGMENTATION
    print("Segmentando partes ...")

    #DensePose predictions
    dp_outputs_iuv = dp_predictor_iuv(image)
    dp_instances_iuv = dp_outputs_iuv["instances"]
    dp_dp_iuv = dp_instances_iuv.pred_densepose[0]
    dp_bbox_iuv = dp_instances_iuv.pred_boxes.tensor[0].numpy()
    dp_coarse_segm = dp_dp_iuv.coarse_segm.numpy()[0]
    dp_fine_segm = dp_dp_iuv.fine_segm.numpy()[0]

    #Most probable class for each pixel
    dp_coarse_mask = dp_coarse_segm.argmax(axis=0)
    dp_fine_mask = dp_fine_segm.argmax(axis=0)

    #Combination of masks to obtain the body parts within the body area
    dp_segm_mask =  dp_fine_mask * dp_coarse_mask

    #Refined segmentation
    dp_segm_mask = smooth_labels(dp_segm_mask)
    dp_segm_mask = correct_anatomical_anomalies(dp_segm_mask)

    #Aligned segmentation
    x1, y1, x2, y2 = dp_bbox_iuv.astype(int)
    h_box, w_box = y2 - y1, x2 - x1
    dp_segm_mask_resized = cv2.resize(dp_segm_mask, (w_box, h_box), interpolation=cv2.INTER_NEAREST)
    dp_segm_mask_full = np.zeros(image.shape[:2])
    dp_segm_mask_full[y1:y2, x1:x2] = dp_segm_mask_resized

    #Extrat vertices id functions
    f_vids = lambda dict_segm_vids: torch.tensor(list(dict_segm_vids.keys()))
    f_vids_part = lambda dict_segm_vids, part: [vid
        for vid, p in dict_segm_vids.items()
            if p == part
    ]

    #Mesh dict
    proj_mesh = project_points(K, vmesh_scan)
    h, w = dp_segm_mask_full.shape[:2]
    dict_segm_vids_mesh = {
        vid: DP_BODY_PARTS[dp_segm_mask_full[v, u]]
        for vid, (u, v) in enumerate(proj_mesh)
            if 0 <= v < h and 0 <= u < w
            and dp_segm_mask_full[v, u] in DP_BODY_PARTS
    }
    dict_segm_vids_mesh = fill_unlabeled_points(dict_segm_vids_mesh, vmesh_scan)
    vmesh = vmesh_scan[f_vids(dict_segm_vids_mesh)]

    #Parts
    visible_dp_parts = set(dict_segm_vids_mesh.values())

    #Star dict
    dict_segm_vids_star = dict(sorted({
        vid: dp_part
        for dp_part in sorted(visible_dp_parts)
            for smpl_part in DP_TO_SMPL[dp_part]
                for vid in smpl_vert_segmentation[smpl_part]
    }.items()))
    vstar = vstar_model[f_vids(dict_segm_vids_star)]

    #ALIGNMENT
    print("Alineando ...")

    #Centroids of each part
    centroids_parts_mesh = [
        vmesh_scan[f_vids_part(dict_segm_vids_mesh, part)].mean(axis=0)
        for part in visible_dp_parts
    ]
    centroids_parts_star = [
        vstar_model[f_vids_part(dict_segm_vids_star, part)].mean(axis=0)
        for part in visible_dp_parts
    ]

    R, t = ransac_kabsch(np.array(centroids_parts_star), np.array(centroids_parts_mesh))
    f_align_vstar = lambda vstar, R=R, t=t: (R @ vstar.T).T + t
    f_vstar_algined = lambda star_model: f_align_vstar(f_vstar(star_model))

    #Export alignment results
    vmesh = vmesh_scan[f_vids(dict_segm_vids_mesh)]
    trimesh.Trimesh(vertices=vmesh, vertex_colors=fixed_coloring_parts(dict_segm_vids_mesh), process=False
    ).export(RESULTPATH+scanid+"-mesh_segm.ply", file_type="ply")
    vstar_aligned = f_align_vstar(vstar)
    trimesh.Trimesh(vertices=vstar_aligned, vertex_colors=fixed_coloring_parts(dict_segm_vids_star), faces=extract_submesh_faces(star_model.f, f_vids(dict_segm_vids_star)), process=False
    ).export(RESULTPATH+scanid+"-star_segm_align.ply", file_type="ply");

    #ADJUSTMENT
    print("Ajustando ...")

    def objective_func(params, star, f_vstar, dict_segm_vids_star, pose, shape, trans, vmesh):
        pose, pose_indices_to_minimize = pose

        i_params = 0
        if pose_indices_to_minimize != None:
            pose[:,pose_indices_to_minimize] = torch.tensor(params[:len(pose_indices_to_minimize)], dtype=torch.float32).unsqueeze(0)
            i_params += len(pose_indices_to_minimize)
        if shape == None:
            shape = torch.tensor(params[i_params:i_params+NBETAS], dtype=torch.float32).unsqueeze(0)
            i_params += NBETAS
        if trans == None:
            trans = torch.tensor(params[i_params:], dtype=torch.float32).unsqueeze(0)
            i_params += NTRANS

        star_model = star(pose, shape, trans)
        vstar = f_vstar(star_model)[f_vids(dict_segm_vids_star)]

        loss = chamfer_dist(vstar, vmesh)
        return loss

    pose_indices_to_minimize = sorted(set(
        idx
        for part in visible_dp_parts
            for idx in BODY_PART_TO_SMPL_POSE_INDICES[part]
    ))

    t0optm = time.time()

    optm_pose = pose.clone()
    optm_shape = shape.clone()
    optm_trans = trans.clone()

    for i in range(1):
        print(f"... iteracion global {i}")

        #Pose adjustment
        print("... ... en pose        ", end="\t -> ", flush=True)
        pose_adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_pose.squeeze(0).numpy()[pose_indices_to_minimize],
            ]),
            args=(star, f_vstar_algined, dict_segm_vids_star,
                (optm_pose, pose_indices_to_minimize), optm_shape, optm_trans,
                vmesh
            ),
            bounds=POSE_BOUNDS[pose_indices_to_minimize],
            method="SLSQP",
            options={"maxiter":5, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {pose_adjuster_result.fun}")
        optm_pose[:,pose_indices_to_minimize] = torch.tensor(pose_adjuster_result.x, dtype=torch.float32).unsqueeze(0)

        #Shape adjustment
        print("... ... en forma       ", end="\t -> ", flush=True)
        shape_adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_shape.squeeze(0).numpy(),
            ]),
            args=(star, f_vstar_algined, dict_segm_vids_star,
                (optm_pose, None), None, optm_trans,
                vmesh
            ),
            bounds=SHAPE_BOUNDS,
            method="SLSQP",
            options={"maxiter":20, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {shape_adjuster_result.fun}")
        optm_shape = torch.tensor(shape_adjuster_result.x, dtype=torch.float32).unsqueeze(0)

        #Pose-Shape adjustment
        print("... ... en pose y forma", end="\t -> ", flush=True)
        adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_pose.squeeze(0).numpy()[pose_indices_to_minimize],
                optm_shape.squeeze(0).numpy(),
            ]),
            args=(star, f_vstar_algined, dict_segm_vids_star,
                (optm_pose, pose_indices_to_minimize), None, optm_trans,
                vmesh
            ),
            bounds=np.concatenate((POSE_BOUNDS[pose_indices_to_minimize], SHAPE_BOUNDS)),
            method="SLSQP",
            options={"maxiter":10, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {adjuster_result.fun}")
        optm_pose[:,pose_indices_to_minimize] = torch.tensor(adjuster_result.x[:len(pose_indices_to_minimize)], dtype=torch.float32).unsqueeze(0)
        optm_shape = torch.tensor(adjuster_result.x[len(pose_indices_to_minimize):len(pose_indices_to_minimize)+NBETAS], dtype=torch.float32).unsqueeze(0)

    #SYNTHETIC EXPERIMENTATION
    print("Ajustando prueba sintetica ...")

    syn_vstar = synt_vstar_model[f_vids(dict_segm_vids_star)]

    optm_synt_pose = pose.clone()
    optm_synt_shape = shape.clone()
    optm_synt_trans = trans.clone()

    for i in range(5):
        print(f"... iteracion global {i}")

        #Pose adjustment
        print("... ... en pose        ", end="\t -> ", flush=True)
        pose_adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_synt_pose.squeeze(0).numpy()[pose_indices_to_minimize],
            ]),
            args=(star, f_vstar, dict_segm_vids_star,
                (optm_synt_pose, pose_indices_to_minimize), optm_synt_shape, optm_synt_trans,
                syn_vstar
            ),
            bounds=POSE_BOUNDS[pose_indices_to_minimize],
            method="SLSQP",
            options={"maxiter":25, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {pose_adjuster_result.fun}")
        optm_synt_pose[:,pose_indices_to_minimize] = torch.tensor(pose_adjuster_result.x, dtype=torch.float32).unsqueeze(0)

        #Shape adjustment
        print("... ... en forma       ", end="\t -> ", flush=True)
        shape_adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_synt_shape.squeeze(0).numpy(),
            ]),
            args=(star, f_vstar, dict_segm_vids_star,
                (optm_synt_pose, None), None, optm_synt_trans,
                syn_vstar
            ),
            bounds=SHAPE_BOUNDS,
            method="SLSQP",
            options={"maxiter":125, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {shape_adjuster_result.fun}")
        optm_synt_shape = torch.tensor(shape_adjuster_result.x, dtype=torch.float32).unsqueeze(0)

        #Pose-Shape adjustment
        print("... ... en pose y forma", end="\t -> ", flush=True)
        adjuster_result = scipy.optimize.minimize(
            fun=objective_func,
            x0=np.concatenate([
                optm_synt_pose.squeeze(0).numpy()[pose_indices_to_minimize],
                optm_synt_shape.squeeze(0).numpy(),
            ]),
            args=(star, f_vstar, dict_segm_vids_star,
                (optm_synt_pose, pose_indices_to_minimize), None, optm_synt_trans,
                syn_vstar
            ),
            bounds=np.concatenate((POSE_BOUNDS[pose_indices_to_minimize], SHAPE_BOUNDS)),
            method="SLSQP",
            options={"maxiter":50, "disp":False, "eps":1e-5, "ftol":1e-6, "xatol":1e-3, "fatol":1e-6}
        )
        print(f"perdida final: {adjuster_result.fun}")
        optm_synt_pose[:,pose_indices_to_minimize] = torch.tensor(adjuster_result.x[:len(pose_indices_to_minimize)], dtype=torch.float32).unsqueeze(0)
        optm_synt_shape = torch.tensor(adjuster_result.x[len(pose_indices_to_minimize):len(pose_indices_to_minimize)+NBETAS], dtype=torch.float32).unsqueeze(0)

    #Export adjustment results
    optm_star_model = star(optm_pose, optm_shape, optm_trans)
    optm_vstar_model_aligned = f_align_vstar(f_vstar(optm_star_model))
    trimesh.Trimesh(vertices=optm_vstar_model_aligned, faces=optm_star_model.f, process=False
    ).export(RESULTPATH+scanid+"-star_optm.ply", file_type="ply");
    trimesh.Trimesh(syn_vstar, vertex_colors=fixed_coloring_parts(dict_segm_vids_star), faces=extract_submesh_faces(synt_star_model.f, f_vids(dict_segm_vids_star)), process=False
    ).export(RESULTPATH+scanid+"-star_synt_segm.ply", file_type="ply");
    optm_synt_star_model = star(optm_synt_pose, optm_synt_shape, optm_synt_trans)
    optm_synt_vstar_model = f_vstar(optm_synt_star_model)
    trimesh.Trimesh(vertices=optm_synt_vstar_model, faces=optm_synt_star_model.f, process=False
    ).export(RESULTPATH+scanid+"-star_synt_optm.ply", file_type="ply");

    optm_poses.append(optm_pose.squeeze(0).numpy())
    optm_shapes.append(optm_shape.squeeze(0).numpy())
    optm_synt_poses.append(optm_synt_pose.squeeze(0).numpy())
    optm_synt_shapes.append(optm_synt_shape.squeeze(0).numpy())

    tsoptm.append(time.time()-t0optm)

print("\n==============================\n")

#Global model
global_optm_pose = torch.tensor(np.mean(optm_poses, axis=0)).unsqueeze(0)
global_optm_shape = torch.tensor(np.mean(optm_shapes, axis=0)).unsqueeze(0)
global_optm_star_model = star(global_optm_pose, global_optm_shape, trans)
global_synt_optm_pose = torch.tensor(np.mean(optm_synt_poses, axis=0)).unsqueeze(0)
global_synt_optm_shape = torch.tensor(np.mean(optm_synt_shapes, axis=0)).unsqueeze(0)
global_synt_optm_star_model = star(global_synt_optm_pose, global_synt_optm_shape, trans)

#Export global model result
trimesh.Trimesh(vertices=vstar_model, faces=star_model.f, process=False
).export(RESULTPATH+"star.ply", file_type="ply")
global_optm_vstar = f_vstar(global_optm_star_model)
trimesh.Trimesh(vertices=global_optm_vstar, faces=global_optm_star_model.f, process=False
).export(RESULTPATH+"star_optm.ply", file_type="ply")
trimesh.Trimesh(vertices=synt_vstar_model, faces=synt_star_model.f, process=False
).export(RESULTPATH+"star_synt.ply", file_type="ply")
global_synt_optm_vstar = f_vstar(global_synt_optm_star_model)
trimesh.Trimesh(vertices=global_synt_optm_vstar, faces=global_optm_star_model.f, process=False
).export(RESULTPATH+"star_synt_optm.ply", file_type="ply")

vregistration = trimesh.load_mesh(DATASETPATH+"/Model.obj", file_type="obj", process=False).vertices

print("... ^^ MODELOS FINALES CREADOS ^^ ...")
print(f"Perdida: {chamfer_dist(global_optm_vstar, vregistration)}")
print(f"Perdida sintetica: {chamfer_dist(global_synt_optm_vstar, synt_vstar_model)}")
print(f"Tiempo de optimización: {sum(tsoptm)}")
print(f"Tiempo total: {time.time()-t0}")
print("\n==============================\n")