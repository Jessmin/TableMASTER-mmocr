from table_inference import Structure_Recognition
from table_recognition.utils import structure_visual


structure_master_config = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/configs/textrecog/master/table_master_ResnetExtract_Ranger_0705.py'
structure_master_ckpt = '/home/zhaohj/Documents/workspace/github_my/TableMASTER-mmocr/output/latest.pth'
fpath = '/home/zhaohj/Documents/dataset/Table/TAL/Table/images/0a1dcef6587d61543efb6bad038eec7f.png'
master_structure_inference = Structure_Recognition(structure_master_config, structure_master_ckpt)
_, result_dict = master_structure_inference.predict_single_file(fpath)
for res_name in result_dict:
    res = result_dict[res_name]
    structure_visual(fpath, res)