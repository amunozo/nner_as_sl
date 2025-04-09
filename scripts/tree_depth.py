import os
import csv
import statistics

def get_trees_from_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    trees = content.strip().split('\n')
    return trees

def calculate_depth(tree):
    max_depth = 0
    current_depth = 0
    for char in tree:
        if char == '(':
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char == ')':
            current_depth -= 1
    return max_depth

def main():
    base_dir = 'data'
    subdirs = ['ace2004', 'ace2005', 'genia', 'nne']
    csv_file = 'tree_depth_statistics.csv'
    headers = ['Folder', 'File', 'Total Trees', 'Min Depth', 'Max Depth', 'Average Depth', 'Median Depth']
    
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for subdir in subdirs:
            dir_path = os.path.join(base_dir, subdir)
            folder_depths = []
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.endswith('.trees'):
                        file_path = os.path.join(root, file)
                        trees = get_trees_from_file(file_path)
                        depths = [calculate_depth(tree) for tree in trees]
                        if depths:
                            total = len(depths)
                            min_depth = min(depths)
                            max_depth = max(depths)
                            avg_depth = sum(depths) / total
                            median_depth = statistics.median(depths)
                            writer.writerow([subdir, file, total, min_depth, max_depth, f"{avg_depth:.2f}", median_depth])
                            folder_depths.extend(depths)
            if folder_depths:
                total = len(folder_depths)
                min_depth = min(folder_depths)
                max_depth = max(folder_depths)
                avg_depth = sum(folder_depths) / total
                median_depth = statistics.median(folder_depths)
                writer.writerow([subdir, 'total', total, min_depth, max_depth, f"{avg_depth:.2f}", median_depth])
    
    print(f"Statistics saved to {csv_file}")

if __name__ == "__main__":
    main()