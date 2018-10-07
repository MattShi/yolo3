import xml.etree.ElementTree as ET
import os
from os import listdir, getcwd
import sys

sets=[('2012', 'train'), ('2012', 'val')]

classes = ["bus", "car", "motorbike", "person"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(rootpath,year, image_id):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(rootpath,year, image_id))
    out_file = open('%s/VOC%s/labels/%s.txt'%(rootpath,year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')




def main():
    if len(sys.argv) < 2:
        print("please input the root of voc dataset")
        return

    voc_root_path = sys.argv[1]

    wd = getcwd()
    for year, image_set in sets:
        if not os.path.exists('%s/VOC%s/labels/' % (voc_root_path,year)):
            os.makedirs('%s/VOC%s/labels/' % (voc_root_path,year))
        image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt' % (voc_root_path,year, image_set)).read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/%s/VOC%s/JPEGImages/%s.jpg\n' % (wd, voc_root_path,year, image_id))
            convert_annotation(voc_root_path,year, image_id)
        list_file.close()

    os.system("cat 2012_train.txt 2012_val.txt > train.txt")
    os.system("cat 2012_train.txt 2012_val.txt > train.all.txt")


if __name__== "__main__":
  main()
