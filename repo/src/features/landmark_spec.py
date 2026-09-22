"""Defines the three controlled landmark representations (spec section 9)
and the facial-landmark subset used instead of all 468 MediaPipe face
points (spec explicitly discourages using all 468 without justification).

MediaPipe Holistic indices (documented, not invented):
  - Left/right hand: 21 landmarks each (x, y, z) -> HAND_LANDMARKS
  - Pose: 33 landmarks (x, y, z, visibility) -> POSE_LANDMARKS
  - Face mesh: 468 landmarks (x, y, z) -> FACE_LANDMARKS

For R2 we keep only the upper-body pose points relevant to signing
(shoulders, elbows, wrists, hips-for-scale -- NOT legs/feet, which carry
no ISL information and only add noise/dimensionality).

For R3 we add a small, named subset of face landmarks (eyebrows, eyes,
mouth outline) which are linguistically relevant in sign languages
(non-manual markers), rather than the full 468-point mesh.
"""

HAND_NUM_LANDMARKS = 21  # per hand, from MediaPipe Hands

# MediaPipe Pose landmark indices (BlazePose topology) relevant to
# upper-body signing posture.
POSE_UPPER_BODY_INDICES = [
    11, 12,  # left/right shoulder
    13, 14,  # left/right elbow
    15, 16,  # left/right wrist
    23, 24,  # left/right hip (used only as a stable scale/torso reference)
]

# MediaPipe Face Mesh indices for non-manual markers relevant to ISL:
# eyebrows, eyes, outer+inner mouth outline. These index numbers follow
# the standard MediaPipe FaceMesh topology and must be verified against
# the installed mediapipe version's canonical face mesh map before the
# first real extraction run -- see the assertion in extract_landmarks.py.
FACE_EYEBROW_INDICES = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
FACE_EYE_INDICES = [33, 133, 160, 159, 158, 157, 173, 263, 362, 387, 386, 385, 384, 398]
FACE_MOUTH_INDICES = [61, 291, 78, 308, 13, 14, 17, 0, 37, 267, 269, 270, 409, 415]

FACE_SELECTED_INDICES = sorted(set(FACE_EYEBROW_INDICES + FACE_EYE_INDICES + FACE_MOUTH_INDICES))

REPRESENTATIONS = {
    "R1": {
        "name": "hands_only",
        "description": "Both hands only (21 landmarks x 2 x 3 coords).",
        "uses_hands": True,
        "uses_pose": False,
        "uses_face": False,
    },
    "R2": {
        "name": "hands_plus_pose",
        "description": "Both hands + upper-body pose (shoulders/elbows/wrists/hips).",
        "uses_hands": True,
        "uses_pose": True,
        "uses_face": False,
    },
    "R3": {
        "name": "hands_plus_pose_plus_face",
        "description": "Both hands + upper-body pose + selected non-manual face landmarks.",
        "uses_hands": True,
        "uses_pose": True,
        "uses_face": True,
    },
}


def feature_dim(representation: str) -> int:
    """Number of scalar features per frame for a given representation,
    using (x, y, z) per hand/pose landmark and (x, y, z) per selected
    face landmark -- documented explicitly so it's checkable, not implicit.
    """
    spec = REPRESENTATIONS[representation]
    dim = 0
    if spec["uses_hands"]:
        dim += 2 * HAND_NUM_LANDMARKS * 3
    if spec["uses_pose"]:
        dim += len(POSE_UPPER_BODY_INDICES) * 3
    if spec["uses_face"]:
        dim += len(FACE_SELECTED_INDICES) * 3
    return dim
