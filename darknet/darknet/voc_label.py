import xml.etree.ElementTree as ET
import sys
import os
from PIL import Image
from shutil import copyfile
from os import listdir, getcwd
from os.path import join

sets=[('2012', 'train'), ('2012', 'val')]

classes_name = ["face"]


def convert(size, box):
    if len(size) != 4 or len(box) != 2:
        return
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


        #tidy filename
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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: dataset_formatt[dlib2yolo3] dataset_folder output_folder')
        exit(1)

    g_format = sys.argv[1]
    g_input = sys.argv[2]
    g_output = sys.argv[3]

    if g_format == "dlib2yolo3":
        process_dlib_2_yolo3(g_input,g_output)
    else:
        print('wrong format[dlib2yolo3]')
