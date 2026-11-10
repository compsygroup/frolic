# Biomechanical constraints: sources for every range and threshold

Module: `frolic/fit/biomech.py` (`ADULT`, `CHILD`, `SOURCES`, `OMEGA_MAX_DPS`). Degrees unless stated. Sign
convention: "rot" > 0 = internal rotation (ankle: inversion); "abd" > 0 = abduction (ankle: foot adduction, toe-in).
Values marked **chosen** are not taken from a reference; they were set for this fit and should be read as such.

References
- AAOS: American Academy of Orthopaedic Surgeons, *Joint Motion: Method of Measuring and Recording* (1965), as reproduced
  in Jones & Bartlett, *Normal Values and Assessments*, tables 1.4 and 1.6, and at goniometer.io/range-of-motion. The
  primary text was not consulted.
- CDC: Soucie et al., Range of motion measurements: reference values and a database for comparison studies,
  *Haemophilia* 2011;17:500-507 (passive goniometry, N = 674, ages 2-69).
- Scherff et al., *J Child Orthop* 2024, table 3 (passive, N = 34, 5-17 y).
- Hemmerich et al., Hip, knee, and ankle kinematics of high range of motion activities of daily living, *J Orthop Res*
  2006 (squatting, kneeling).
- Range of movements of lower limb joints in cross-legged sitting posture, *J Arthroplasty* 2008 (PubMed 18358387).
- Mossberg & Smith, Axial rotation of the knee in women, *JOSPT* 1983; WikiMSK, Knee Biomechanics.
- Rajagopal et al., Full-body musculoskeletal model for muscle-driven simulation of human gait, *IEEE TBME* 2016
  (OpenSim coordinate ranges).
- Keller et al., From skin to skeleton (SKEL / BSM), *SIGGRAPH Asia* 2023 (pose limits).
- Bogo et al., SMPLify, *ECCV* 2016 (capsule interpenetration, bending prior); Pavlakos et al., SMPL-X, *CVPR* 2019.
- Rempe et al., HuMoR, *ICCV* 2021 (contact heights); Shin et al., WHAM, *CVPR* 2024 (contact velocity sigmoid).
- Pan et al., *Sci Rep* 2025 (knee angular velocity in gait); Fleisig, *ISBS* 2010 (pitching elbow extension).
- Grood & Suntay, A joint coordinate system for the clinical description of three-dimensional motions, *J Biomech Eng*
  1983 (Euler order).

## Joint ranges (adult | child)

Child ranges apply fully when the Anny age slider is at the "child" anchor (1/3) and blend linearly to the adult
ranges at `fit.biomech.child_age_hi` (0.45).

| DOF | adult | child | source |
|---|---|---|---|
| hip flex / ext | 130 / 30 | 140 / 30 | flex: CDC 20-44 y 133.8 F / 130.4 M, CDC 2-8 y 140.8 F; ext: AAOS 30 (CDC 2-8 y 26-28) |
| hip abd / add | 60 / 30 | same | add: AAOS 30 (Rajagopal hip adduction [-50, 30]); abd 60: cross-legged sitting needs up to 57 (mean 39), AAOS active 45 |
| hip IR / ER | 45 / 60 | 50 / 60 | IR: AAOS 45, child Scherff prone 50; ER 60: cross-legged sitting needs up to 58 (mean 49), AAOS active 45, Scherff supine 55 |
| knee flex | 160 | 160 | loaded postures exceed active ROM: squatting heels-up 157 +- 6 (Hemmerich 2006), cross-legged 126-142; active CDC 20-44 y 141.9 F, 2-8 y 152.6 F |
| knee hyperextension | 5 | 10 | CDC pooled mean 1.8 (SD 2.9), mean + 1 SD; child: Scherff 4 (SD 4), mean + 1.5 SD; CDC 2-8 y F 5.4 (CI to 6.9) |
| knee varus / valgus | 10 / 10 | same | **chosen** (no reference found) |
| knee IR / ER | 25 / 45, scaled by flexion (20 % at 0 deg, full from 30 deg) | same | 25 IR / 45 ER at 30-90 deg flexion, passive (WikiMSK); Mossberg & Smith 1983: total 35-44 deg at 70-110 deg flexion, ER about 2x IR. The flexion scaling is a **chosen** screw-home approximation |
| ankle dorsi / plantar | 40 / 50 | 40 / 65 | dorsi 40: heels-down squat needs 38.5 +- 5.9 (Hemmerich 2006), AAOS active 20; plantar: AAOS 50, CDC 2-8 y F 67.1; Rajagopal ankle [-40, 30] |
| ankle inversion / eversion | 35 / 15 | same | AAOS |
| foot ab / adduction (toe-in / out vs shank) | 25 / 25 | same | **chosen** between Rajagopal subtalar +-20 and SKEL +-45 |
| shoulder flex / ext | 180 / 60 | same | AAOS |
| shoulder abd / add | 180 / 40 | same | AAOS |
| shoulder IR / ER | 70 / 90 | same | AAOS |
| elbow flex | 150 | 150 | AAOS 150; CDC 2-8 y 152.9 F |
| elbow off-axis (carrying-angle plane) | +-25 | same | **chosen** (misalignment of the Anny hinge plane and the anatomical plane) |
| elbow hyperextension | 5 | 10 | CDC 20-44 y F 4.7; child CDC 2-8 y F 6.8, 9-19 y F 6.4 |
| forearm pronation / supination | +-85 about the rest twist | +-90 | AAOS 80 / 80; CDC 2-8 y pron 84.6 / sup 93.7 F; the rest-pose neutral is unresolved, so the range is symmetric |
| wrist flex / ext | +-80 | same | AAOS 80 / 70 (palm side unresolved, symmetric with the larger value) |
| wrist deviation | +-30 | same | AAOS radial 20 / ulnar 30 |
| wrist twist | +-85 | same | **chosen**: Anny puts part of the forearm pronation / supination on the wrist bone, so the forearm range is used |
| thoracolumbar flex / ext | 80 / 25 | same | AAOS |
| thoracolumbar lateral bend / rotation | 45 / 45 | same | AAOS; the reproductions disagree (45 / 35 vs 35 / 45), the larger value is used |
| spine per bone (spine05 / 03 / 01) | <= 0.6 x chain range | same | **chosen** (Anny poses three spine bones) |
| cervical flex / ext / lateral / rotation (neck01 + head) | 45 / 45 / 45 / 60 | same | AAOS |
| clavicle (sternoclavicular), all DOFs | +-36 | same | SC-joint ROM not found; SKEL scapula abduction limit +-36 is used |

Child values for the spine, wrist and shoulder rotation were not found; adult values are used.

## Terms (how each constraint is modelled)

| term | implementation | basis |
|---|---|---|
| joint limits | box limits per anatomical coordinate; intrinsic Euler X-Z-Y (flexion, ab/adduction, axial rotation) for hip, knee, ankle, elbow, wrist and clavicle; swing-twist for the shoulder (Euler X-Z-Y has a gimbal lock at 90 deg abduction) and the spine / neck chains; Huber hinge with a 5 deg knee | per-coordinate ranges as in OpenSim (Rajagopal 2016) and SKEL / BSM (Keller 2023); SMPLify (Bogo 2016) bending prior on knee / elbow hyperextension; Grood & Suntay 1983 order |
| self-collision | capsules for thigh (distal 70 %), shin, foot, forearm and two torso lines; radius = 0.85 x the median rest-mesh distance of the segment (**chosen** scale); Huber on the overlap depth (1 cm knee). Arm pairs are off by default (crossed arms and a hand at the chin were resolved by elbow jumps); foot pairs on | SMPLify 2016 capsule interpenetration term |
| knee rotation coupling | axial range scaled with flexion | screw-home mechanism (WikiMSK; Mossberg & Smith 1983) |
| foot contact / no-skating | sole points (heels, toes) kept above z = -1 cm; horizontal velocity penalised when in contact: contact = sigmoid((4 cm - z) / 1 cm) x 1 / (1 + exp(5 (v - 0.3) / 0.3)) | HuMoR (Rempe 2021): contact heights 8 cm (joints) / 4 cm (toes); WHAM (Shin 2024): velocity sigmoid with v_t = 1 cm per frame at 30 fps = 0.3 m/s, alpha = 5 |
| angular velocity | soft limit on the per-bone rotation between consecutive frames: 400 deg/s lower limb, trunk and neck; 700 deg/s upper limb (**chosen**) | knee peak in gait 399 +- 72 deg/s (Pan 2025); pitching elbow extension 2450 deg/s (Fleisig 2010) is the extreme the upper-limb value stays well below |
| foot yaw velocity | soft limit 360 deg/s on the heel-to-toe yaw | **chosen** (below the 60 deg per frame flip metric at 15 Hz) |
| foot heading vs knee plane | tibial internal rotation + foot adduction in [-45, 35] deg | sum of the knee rotation range (ER 45 / IR 25) and the foot ab/adduction allowance (**chosen**) |
| bone-length constancy | by construction (one Anny skeleton per person, segment lengths fixed over time) | HuMoR skeleton term; Chen et al., *IEEE TCSVT* 2021 (one bone length per sequence) |
| shape pass | the body shape comes from an unconstrained pass; a second, constrained pass fits the pose only | constraints should change pose, not body size |

Not used: PosePrior (Akhter & Black 2015) pose-conditioned limits (needs their occupancy tables and joint conventions);
VPoser / HuMoR priors (SMPL-specific, not Anny). Anny itself ships no joint limits.

## Initialisation

| item | implementation | basis |
|---|---|---|
| hinge-consistent leg initialisation | per frame: thigh swing to the tracked hip-knee direction; twist chosen so the knee's flexion axis is normal to the hip-knee-ankle plane (axis averaged over +-4 frames, weighted by sin(knee angle)); knee = pure flexion; frames without a full leg copy the nearest frame | SMPLify-style analytic initialisation; without it the optimiser can bend knees about the varus axis with a compensating hip twist |
| root rotation | continuous unit-quaternion track (hemisphere-aligned, gaps filled along the short arc); the fit optimises a small delta on top of it and smooths the geodesic root acceleration | avoids 2 pi jumps of the axis-angle root when the heading crosses +-180 deg |
