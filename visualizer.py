import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import os

def visualizer(fileName, outputName):
    with open(fileName) as f:
        
        #Transforms the input into a grid for example, [["FC","VC"],["VC","FC"]]
        grid = [line.strip().split("\t") for line in f] 

        # Assuming the images are in images directory and named 'FC.png', 'VC.png', etc.
        path_to_images = 'images/'
        
        fig, axs = plt.subplots(len(grid), len(grid[0]), figsize=(10, 10)) 
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        for ax in axs.flatten():
            ax.axis('off')

        for i, row in enumerate(grid):
            for j, img_code in enumerate(row):
                
                img_path = f"{path_to_images}{img_code}.jpg" 
                img = mpimg.imread(img_path) 
                axs[i, j].imshow(img) 

        plt.savefig(outputName)
        plt.close(fig)

# visualizer("output.txt", "output.png")

def list_files_in_directory(directory_path):
    try:
        # Get the list of all entries in the directory
        entries = os.listdir(directory_path)
        
        # Filter out the entries to include only files
        files = [entry for entry in entries if os.path.isfile(os.path.join(directory_path, entry))]
        
        return files
    except FileNotFoundError:
        print("The specified directory was not found.")
        return []
    except PermissionError:
        print("Permission denied to access the specified directory.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


pathFolder = os.path.dirname(os.path.abspath(__file__))
plotsFolder = os.path.join(pathFolder, "plotTable")
outputFolder = os.path.join(pathFolder, "plotTableImages")

plotFiles = list_files_in_directory(plotsFolder)

for file in plotFiles:
    outputName = file.split(".")[0]
    visualizer(os.path.join(plotsFolder, file), os.path.join(outputFolder, f"{outputName}.png"))

# print("Files in directory:", files)