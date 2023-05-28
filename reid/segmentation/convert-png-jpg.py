import os

folder_path = "/home/rdehghani/data/market1501/images"


for filename in os.listdir(folder_path):
    print(filename)
    exit(0)
    
    #if filename.endswith(".jpg.jpg"):
    if filename.endswith(".jpg.png"):
        new_filename = filename[:-8] + ".jpg"
        
        
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
