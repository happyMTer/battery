from PIL import Image

image = Image.open('./result/model_structure/basic_gai_00.png')
width, height = image.size

image  = image.convert('RGBA')
data = image.getdata()

new_data = []
for item in data:
   if item[0]==255 and item[1]==255 and item[2]==255:
       new_data.append((255, 255, 255, 0))  # 透明
   else:
       new_data.append(item)


image.putdata(new_data)

image.save('./result/model_structure/basic_gai_final.png', "PNG")