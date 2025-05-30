import os

problems_folder = "Problems"
folder_names = []

# Check if the "Problems" folder exists
if os.path.exists(problems_folder) and os.path.isdir(problems_folder):
    # Iterate through all items in the "Problems" folder
    for item in os.listdir(problems_folder):
        # Check if the item is a directory
        item_path = os.path.join(problems_folder, item)
        if os.path.isdir(item_path):
            folder_names.append(item)

    # Write the folder names to a text file
    with open("problems_folder_list.txt", "w") as f:
        for name in folder_names:
            f.write(name + "\n")

    print("Folder names inside 'Problems' written to problems_folder_list.txt")
else:
    print("The 'Problems' folder does not exist.")