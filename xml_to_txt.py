import xml.dom.minidom
import os
import glob
from pathlib import Path

'''
this file generats trainimage list, landmark ground truth labeling files from xml file:
this xml file is obtained from 'ImgLab' Annotation toll
'''

input = '/home/chandra/Documents/EAST-master-py36/Samples-20210623T075559Z-001/Samples'
##xml_files = glob.glob(os.path.join('./img/', '*.xml'))
xml_files = glob.glob(os.path.join(input, '*.xml'))
# print(xml_files)
xml_list=[]
for xml_file in xml_files:
    print(os.path.basename(xml_file))
    xml_list.append(xml_file)
    # try:
    doc = xml.dom.minidom.parse(xml_file)
    # print(doc)

    name = doc.getElementsByTagName('path')[0]
    path = name.childNodes[0].nodeValue
    path = path.split('\\')[-1]
    print(path)

    ##fname = './img/{}'.format(path)
    fname = '{}/{}'.format(input,path)
    if not os.path.isfile(fname):  # can't find corresponding jpg file
        print('Error: can not find {} file'.format(fname))
        continue
    # print('cur filename :', fname)

    tmp_name = Path(fname)
    # print(tmp_name)
    tmp_name = tmp_name.with_suffix('.txt')
    # print("Done")
    f = open(tmp_name, 'w')

    #get size information
    size_info = doc.getElementsByTagName('size')[0]
    imw = float(size_info.getElementsByTagName('width')[0].childNodes[0].nodeValue)
    imh = float(size_info.getElementsByTagName('height')[0].childNodes[0].nodeValue)
    class_id = 0
    objects = doc.getElementsByTagName('object')
    for object in objects:
        obj_name = object.getElementsByTagName('name')[0].childNodes[0].nodeValue
        if obj_name == 'TEXT':
            class_id = 0
        bndBox = object.getElementsByTagName('bndbox')[0]
        l = float(bndBox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
        t = float(bndBox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
        r = float(bndBox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
        b = float(bndBox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)

        f.write('%d,%d,%d,%d,%d,%d,%d,%d,%s\n'%(l, t, r, t, r, b, l, b, "0"))
    f.close()
    # except:
    #     continue
# xmls=[]
# for i in xml_list:
#     k= os.path.basename(i)
#     k.split("/")
#     xmls.append(k)

# for xml_file in xmls:
#     if xml_file in os.listdir(input):
#         os.remove(os.path.join(input, xml_file))
# print("Done")