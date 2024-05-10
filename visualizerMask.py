import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys

def visualizer():
    with sys.stdin as f:
        #Transforms the input into a grid for example, [["FC","VC"],["VC","FC"]]
        grid = [line.strip().split("\t") for line in f] 

        # Assuming the images are in images directory and named 'FC.png', 'VC.png', etc.
        path_to_images = 'images/'

        fig, axs = plt.subplots(len(grid), len(grid[0]), figsize=(5, 5)) 

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        for ax in axs.flatten():
            ax.axis('off')

        for i, row in enumerate(grid):
            for j, img_code in enumerate(row):
                # if img_code == "True":
                #     print("True")
                # elif img_code == "False":
                #     print("False")
                # else:
                #     print("None")
                    
                    
                img_path = f"{path_to_images}{img_code}.jpg" 
                img = mpimg.imread(img_path) 
                axs[i, j].imshow(img) 

        plt.savefig('outputMask.png')
visualizer()