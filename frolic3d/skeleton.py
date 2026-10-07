"""The 17-joint body layout shared by every stage, and its mappings to the detector, SAM 3D Body and Anny."""

from __future__ import annotations

JOINT_NAMES: list[str] = [
    "pelvis",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
    "toe_l",
    "toe_r",
    "neck",
    "head",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
]
J = {n: i for i, n in enumerate(JOINT_NAMES)}
N_JOINTS = len(JOINT_NAMES)

# Halpe-26 keypoint index of each joint (RTMPose body7 / halpe26 output).
HALPE26: list[int] = [19, 11, 12, 13, 14, 15, 16, 20, 23, 18, 17, 5, 6, 7, 8, 9, 10]
HALPE26_TORSO: list[int] = [5, 6, 11, 12, 18, 19]
HALPE26_HIP, HALPE26_NECK = 19, 18

# Right/left mirror of each joint (for left-right swap tests).
MIRROR: list[int] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 9, 10, 12, 11, 14, 13, 16, 15]

# Bones used for limb-length constraints and single-view lifting: (proximal, distal).
LIMBS: list[tuple[int, int]] = [
    (J["pelvis"], J["hip_l"]),
    (J["pelvis"], J["hip_r"]),
    (J["hip_l"], J["knee_l"]),
    (J["hip_r"], J["knee_r"]),
    (J["knee_l"], J["ankle_l"]),
    (J["knee_r"], J["ankle_r"]),
    (J["neck"], J["shoulder_l"]),
    (J["neck"], J["shoulder_r"]),
    (J["shoulder_l"], J["elbow_l"]),
    (J["shoulder_r"], J["elbow_r"]),
    (J["elbow_l"], J["wrist_l"]),
    (J["elbow_r"], J["wrist_r"]),
]
LIMBS_WEAK: list[tuple[int, int]] = [
    (J["pelvis"], J["neck"]),
    (J["neck"], J["head"]),
    (J["ankle_l"], J["toe_l"]),
    (J["ankle_r"], J["toe_r"]),
]

# Segments whose length is used for body-size statistics: name -> (joint a, joint b).
SEGMENTS: dict[str, tuple[int, int]] = {
    "thigh_l": (J["hip_l"], J["knee_l"]),
    "thigh_r": (J["hip_r"], J["knee_r"]),
    "shin_l": (J["knee_l"], J["ankle_l"]),
    "shin_r": (J["knee_r"], J["ankle_r"]),
    "upperarm_l": (J["shoulder_l"], J["elbow_l"]),
    "upperarm_r": (J["shoulder_r"], J["elbow_r"]),
    "forearm_l": (J["elbow_l"], J["wrist_l"]),
    "forearm_r": (J["elbow_r"], J["wrist_r"]),
    "shoulders": (J["shoulder_l"], J["shoulder_r"]),
    "hips": (J["hip_l"], J["hip_r"]),
    "trunk": (J["pelvis"], J["neck"]),
}

# SAM 3D Body (MHR, 70 keypoints): joint -> MHR keypoint index.
MHR70_BODY: dict[int, int] = {
    J["hip_l"]: 9,
    J["hip_r"]: 10,
    J["knee_l"]: 11,
    J["knee_r"]: 12,
    J["ankle_l"]: 13,
    J["ankle_r"]: 14,
    J["neck"]: 69,
    J["shoulder_l"]: 5,
    J["shoulder_r"]: 6,
    J["elbow_l"]: 7,
    J["elbow_r"]: 8,
    J["wrist_l"]: 62,
    J["wrist_r"]: 41,
}
MHR70_HIPS = (9, 10)
MHR70_SHOULDERS = (5, 6)
MHR70_WRIST_R, MHR70_WRIST_L = 41, 62
# Halpe-26 index -> MHR keypoint (for relative joint depths in the single-view lift and the MHR fit).
HALPE26_TO_MHR70: dict[int, int] = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 62, 10: 41, 11: 9, 12: 10, 13: 11, 14: 12,
    15: 13, 16: 14, 18: 69, 20: 15, 21: 18, 22: 16, 23: 19, 24: 17, 25: 20,
}  # fmt: skip
# MHR face / foot surface keypoints -> SMPL-X vertex (Anny uses the SMPL-X topology): nose, eyes, ears, toes, heels.
MHR70_SURFACE: list[tuple[int, int]] = [
    (0, 9120), (1, 9448), (2, 9929), (3, 6), (4, 616),
    (15, 5770), (16, 5780), (17, 8846), (18, 8463), (19, 8474), (20, 8635),
]  # fmt: skip
MHR70_SOLE = slice(5, 11)  # rows of MHR70_SURFACE that lie on the soles (toes, heels)
MHR70_FEET: list[int] = [13, 14, 15, 16, 17, 18, 19, 20]

# Anny ("default" rig) bone point of each joint: (bone, head|tail, weight). The pelvis is the hip midpoint.
ANNY_POINTS: dict[int, tuple[str, str, float]] = {
    J["hip_l"]: ("upperleg01.L", "h", 1.0),
    J["hip_r"]: ("upperleg01.R", "h", 1.0),
    J["knee_l"]: ("lowerleg01.L", "h", 1.0),
    J["knee_r"]: ("lowerleg01.R", "h", 1.0),
    J["ankle_l"]: ("foot.L", "h", 1.0),
    J["ankle_r"]: ("foot.R", "h", 1.0),
    J["toe_l"]: ("toe1-1.L", "h", 0.3),
    J["toe_r"]: ("toe5-1.R", "h", 0.3),
    J["neck"]: ("neck01", "h", 0.7),
    J["head"]: ("head", "t", 0.5),
    J["shoulder_l"]: ("upperarm01.L", "h", 1.0),
    J["shoulder_r"]: ("upperarm01.R", "h", 1.0),
    J["elbow_l"]: ("lowerarm01.L", "h", 1.0),
    J["elbow_r"]: ("lowerarm01.R", "h", 1.0),
    J["wrist_l"]: ("wrist.L", "h", 1.0),
    J["wrist_r"]: ("wrist.R", "h", 1.0),
}
ANNY_POSED_BONES: list[str] = [
    "spine05", "spine03", "spine01", "neck01", "head", "clavicle.L", "clavicle.R",
    "upperarm01.L", "upperarm01.R", "lowerarm01.L", "lowerarm01.R", "wrist.L", "wrist.R",
    "upperleg01.L", "upperleg01.R", "lowerleg01.L", "lowerleg01.R", "foot.L", "foot.R",
] + [f"finger{f}-{k}.{sd}" for sd in "LR" for f in range(1, 6) for k in range(1, 4)]  # fmt: skip
ANNY_SEGMENT_LABELS: list[str] = [
    "measure-upperarm-length-incr",
    "measure-lowerarm-length-incr",
    "measure-upperleg-height-incr",
    "measure-lowerleg-height-incr",
    "measure-napetowaist-dist-incr",
    "measure-waisttohip-dist-incr",
    "measure-neck-height-incr",
]
