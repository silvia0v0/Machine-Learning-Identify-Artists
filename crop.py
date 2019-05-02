import cv2
import json

paintings = {'Vincent_van_Gogh_': 877, 'Edgar_Degas_': 702, 'Pablo_Picasso_': 439, 'Pierre-Auguste_Renoir_': 336, 'Paul_Gauguin_':311, 'Francisco_Goya_':291, 'Rembrandt_':262, 'Alfred_Sisley_':259, 'Titian_':255, 'Marc_Chagall_':239}
fn_data = {}
for painter in paintings:
    fn_data[painter] = {}
    for n in range(1, paintings[painter] + 1):
        path = './resized/resized/' + painter + str(n) + '.jpg'
        print(path)
        img = cv2.imread(path)
        shape = img.shape
        length = shape[1] // 128
        width = shape[0] // 128
        print(length, width)
        num = 0
        for l in range(length):
            for w in range(width):
                # cropped = img[w*128:(w+1)*128, l*128:(l+1)*128]
                # cv2.imwrite('./resized/resized/cropped/' + painter + str(n) + '_' + str(num)+'.jpg', cropped)
                num += 1
        fn_data[painter][n] = num

    with open('fn_data.json', 'w') as f:
        json.dump(fn_data, f)
