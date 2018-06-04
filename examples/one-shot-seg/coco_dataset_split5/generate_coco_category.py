

def generate_id2trainid(_coco):
    new_dict = {}
    for idx,data in enumerate(_coco.cats.items()):
        key,value = data #start from 1!!!
        new_dict[key] = idx+1

    return new_dict

from pycocotools.coco import COCO
_coco = COCO("/data2/dataset/annotations/instances_train2014.json")
catId_to_ascendorder = generate_id2trainid(_coco)
name2trainid = {value['name']:catId_to_ascendorder[value['id']] for (key,value) in  _coco.cats.items()}

for key,value in name2trainid.items():
    print("'{}':{},".format(key,value))

print("catId_to_ascendorder length: {}".format(len(catId_to_ascendorder)))

for key,value in catId_to_ascendorder.items():
    print("{}:{},".format(key,value))

pass

