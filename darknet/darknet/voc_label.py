import xml.etree.ElementTree as ET
import sys
import os
from PIL import Image
from shutil import copyfile
from os import listdir, getcwd
from os.path import join


sets=[('2012', 'train'), ('2012', 'val')]

classes_name = ["face","bicycle","bus", "car",  "motorbike", "person"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convertdlib2yolo3(size, box):
    if len(size) != 2 or len(box) != 4:
        return

    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2]/2.0)
    y = (box[1] + box[3]/2.0)

    x = x*dw
    w = box[2]*dw
    y = y*dh
    h = box[3]*dh
    return (x,y,w,h)


def trans_dlib_2_yolo3(treenode,outfile,infoler,outfolder,clsname):
    clsid = classes_name.index(clsname)
    if clsid < 0:
        return

    out_datasefile = open(outfile, 'w')

    if not os.path.exists("%s/labels" %(outfolder)):
        os.mkdir("%s/labels" %(outfolder))

    if not os.path.exists("%s/JPEGImages" % (outfolder)):
            os.mkdir("%s/JPEGImages" % (outfolder))

    root = treenode.getroot()
    for img in root.iter('image'):
        imgid = img.get('file')
        imgfilebasename = os.path.basename(imgid)
        imgfilename, imgextension = os.path.splitext(imgfilebasename)
        imgfilename = imgfilename.replace("jpg",'_')
        imgfilename = imgfilename.replace("JPG", '_')
        imgfilename = imgfilename.replace("jpeg", '_')
        imgfilename = imgfilename.replace("JPEG", '_')
        imgfilename = imgfilename.replace("png", '_')
        imgfilename = imgfilename.replace("PNG", '_')

        imgfilenewname = imgfilename + imgextension

        cpfile_from = ("%s/%s" % (infoler,imgid))
        cpfile_to = ("%s/JPEGImages/%s" % (outfolder,imgfilenewname))
        copyfile(cpfile_from,cpfile_to)

        out_labelfile = open('%s/labels/%s.txt' % (outfolder,imgfilename), 'w')
        out_datasefile.write('%s\n' %(cpfile_to))

        for imgbox in img.iter('box'):

            if imgbox.get('ignore') == '1':
                continue

            box = (float(imgbox.get('left')), float(imgbox.get('top')),
                    float(imgbox.get('width')), float(imgbox.get('height')))

            img = Image.open('%s/%s' %(infoler,imgid))
            box_cvt = convertdlib2yolo3(img.size,box)
            out_labelfile.write(str(clsid) + " " + " ".join([str(a) for a in box_cvt]) + '\n')


def process_dlib_2_yolo3(infoler,outfolder):
    tree_train = ET.parse(infoler+"/training.xml")
    tree_test = ET.parse(infoler + "/testing.xml")
    trans_dlib_2_yolo3(tree_train,outfolder+"/train.txt",infoler,outfolder,"face")
    trans_dlib_2_yolo3(tree_test,outfolder+"/test.txt",infoler,outfolder,"face")




def convert_annotation_voc_yolo3(infoler,outfolder,year, image_id):
    in_file = open('%s/VOC%s/Annotations/%s.xml'%(infoler,year, image_id))
    out_file = open('%s/VOC%s/labels/%s.txt'%(infoler,year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes_name or int(difficult) == 1:
            continue
        cls_id = classes_name.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def trans_voc_2_yolo3(infoler,outfolder):
    for year, image_set in sets:
        if not os.path.exists('%s/VOC%s/labels/' % (infoler,year)):
            os.makedirs('%s/VOC%s/labels/' % (infoler,year))
        image_ids = open('%s/VOC%s/ImageSets/Main/%s.txt' % (infoler,year, image_set)).read().strip().split()
        list_file = open('%s/%s_%s.txt' % (outfolder,year, image_set), 'w')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg\n' % (infoler, year, image_id))
            convert_annotation_voc_yolo3(infoler,outfolder,year, image_id)
        list_file.close()

def process_voc_2_yolo3(infoler,outfolder):
    trans_voc_2_yolo3(infoler,outfolder)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: dataset_formatt[dlib2yolo3] dataset_folder output_folder')
        exit(1)

    g_format = sys.argv[1]
    g_input = sys.argv[2]
    g_output = sys.argv[3]

    if g_format == "dlib2yolo3":
        process_dlib_2_yolo3(g_input,g_output)
    if g_format == "voc2yolo3":
        process_voc_2_yolo3(g_input, g_output)
        print('wrong format[dlib2yolo3]')
