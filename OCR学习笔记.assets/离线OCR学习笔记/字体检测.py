from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype("fonts/simfang.ttf", 24)
img = Image.new("RGB", (200, 50), (255, 255, 255))
draw = ImageDraw.Draw(img)
draw.text((10, 10), "仿宋测试", font=font, fill=(0, 0, 0))
img.save("test_font.png")