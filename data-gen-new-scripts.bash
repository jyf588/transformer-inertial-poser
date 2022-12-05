# Copyright (c) Meta, Inc. and its affiliates.
# Copyright (c) Stanford University

SAVE_PARENT="./data"
AMASS_PARENT="./data/source"
DATA_V_TAG="v1"
N_PROC="7"

python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_AMASS_CMU_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/AMASS_CMU/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_KIT_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/KIT/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_Eyes_Japan_Dataset_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/Eyes_Japan_Dataset --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_HUMAN4D_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/HUMAN4D --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_ACCAD_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/ACCAD --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_DFaust_67_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/DFaust_67/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_HumanEva_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/HumanEva --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_MPI_Limits_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/MPI_Limits/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_MPI_mosh_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/MPI_mosh/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_SFU_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/SFU/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_Transitions_mocap_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/Transitions_mocap/ --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_TotalCapture_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/TotalCapture --n_proc ${N_PROC}
python data-gen-and-viz-bullet-new.py --save_dir ${SAVE_PARENT}/syn_DanceDB_${DATA_V_TAG} --src_dir ${AMASS_PARENT}/DanceDB --n_proc ${N_PROC}
